"""
engram_mastery.grading.worker
==============================
Async grading job worker — polls the DB queue and processes pending jobs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..core.mastery_service import (
    MasteryRepository,
    apply_grading_result,
    apply_short_question_grading_result,
)
from ..core.types import (
    CognitiveLevel,
    EngramType,
    LongQuestionContent,
    QuizContent,
)
from ..storage.vector_store import (
    EmbeddingProvider,
    VectorStore,
    index_answer,
    retrieve_past_answers,
)
from .ai_grading import (
    GradingPipelineInput,
    ShortQuestionGradingPipelineInput,
    run_grading_pipeline,
    run_short_question_grading_pipeline,
)
from .context_retrieval import retrieve_grading_context

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BATCH_SIZE = 5
POLL_INTERVAL = 5  # seconds


# ---------------------------------------------------------------------------
# Job payload
# ---------------------------------------------------------------------------


@dataclass
class GradingJobPayload:
    job_id: str
    attempt_id: str
    engram_id: str
    user_id: str
    topic: str
    raw_answer: str
    question: LongQuestionContent
    target_cognitive_level: int
    bubble_id: str = ""  # keys the note's cached-retrieval store
    note_ids: list[str] = field(default_factory=list)


@dataclass
class ShortQuestionGradingJobPayload:
    """Short-question analogue of GradingJobPayload. Carries the whole
    QuizContent plus the student's raw answers keyed by question_index —
    one grading_jobs row grades every sub-question of the attempt."""

    job_id: str
    attempt_id: str
    engram_id: str
    user_id: str
    topic: str
    content: QuizContent
    answers: dict[int, str]  # question_index -> raw_answer
    target_cognitive_level: int
    bubble_id: str = ""  # keys the note's cached-retrieval store
    note_ids: list[str] = field(default_factory=list)


# A hydrated job is one of the two payload shapes; run_worker_batch dispatches
# on the concrete type.
GradingPayload = "GradingJobPayload | ShortQuestionGradingJobPayload"


# ---------------------------------------------------------------------------
# Worker loop protocol (implement against your DB)
#
# fetch_pending_jobs / get_grading_context on MasteryRepository (added to
# sqlite_repository.NoteEngramRepository) back this. Each pending job dict
# has keys: job_id, attempt_id, attempts, priority, created_at.
#
# hydrate_job takes BOTH job_id and attempt_id — the job_id is needed to
# populate GradingJobPayload.job_id, and the caller (run_worker_batch)
# already has both from the fetch_pending_jobs row, so there's no reason to
# make hydrate_job re-derive job_id from attempt_id via a second lookup.
# ---------------------------------------------------------------------------


class WorkerLoop(Protocol):
    def fetch_pending_jobs(self, limit: int) -> list[dict]: ...
    def hydrate_job(
        self, job_id: str, attempt_id: str
    ) -> "Optional[GradingJobPayload | ShortQuestionGradingJobPayload]": ...
    def mark_processing(self, job_id: str) -> None: ...
    def mark_done(self, job_id: str) -> None: ...
    def mark_failed(self, job_id: str, error: str) -> None: ...


class SQLiteWorkerLoop:
    """WorkerLoop implementation against MasteryRepository (NoteEngramRepository).

    grading_jobs rows are created for both long_question and short_question
    attempts (see mastery_service.submit_long_question /
    submit_short_question), so hydrate_job dispatches on the engram type and
    returns the matching payload shape. Any other type (mcq/flashcard, which
    are graded synchronously and never enqueue a job) or a missing engram is
    a hard failure for that job rather than a silent skip.
    """

    def __init__(self, repo: MasteryRepository) -> None:
        self._repo = repo

    def fetch_pending_jobs(self, limit: int) -> list[dict]:
        return self._repo.fetch_pending_grading_jobs(limit=limit)

    def hydrate_job(
        self, job_id: str, attempt_id: str
    ) -> "Optional[GradingJobPayload | ShortQuestionGradingJobPayload]":
        # engram type isn't on the attempt row, so peek at it first to pick
        # the right context/content join. get_grading_context is a superset
        # of get_short_question_grading_context (it also LEFT JOINs the long
        # answer), so it's a safe way to resolve engram_id for either type.
        base = self._repo.get_grading_context(attempt_id)
        if base is None:
            logger.warning(
                "hydrate_job: no engram_attempts row for attempt %s", attempt_id
            )
            return None

        engram = self._repo.get_engram(base["engram_id"])
        if engram is None:
            logger.warning(
                "hydrate_job: engram %s (attempt %s) not found",
                base["engram_id"],
                attempt_id,
            )
            return None

        if engram.type == EngramType.LONG_QUESTION:
            return self._hydrate_long_question(job_id, attempt_id, base, engram)
        if engram.type == EngramType.SHORT_QUESTION:
            return self._hydrate_short_question(job_id, attempt_id, engram)

        logger.warning(
            "hydrate_job: engram %s is type=%s — grading_jobs are only created "
            "for long_question/short_question attempts (mcq/flashcard grade "
            "synchronously)",
            base["engram_id"],
            engram.type,
        )
        return None

    def _hydrate_long_question(
        self, job_id: str, attempt_id: str, ctx: dict, engram
    ) -> Optional[GradingJobPayload]:
        if not isinstance(engram.content, LongQuestionContent):
            logger.warning(
                "hydrate_job: engram %s content is not LongQuestionContent",
                ctx["engram_id"],
            )
            return None

        return GradingJobPayload(
            job_id=job_id,
            attempt_id=attempt_id,
            engram_id=ctx["engram_id"],
            user_id=ctx["user_id"],
            topic=ctx.get("topic") or "",
            raw_answer=ctx.get("raw_answer") or "",
            question=engram.content,
            target_cognitive_level=ctx["target_cognitive_level"],
            bubble_id=engram.bubble_id or "",
            note_ids=[ctx["note_id"]] if ctx.get("note_id") else [],
        )

    def _hydrate_short_question(
        self, job_id: str, attempt_id: str, engram
    ) -> Optional[ShortQuestionGradingJobPayload]:
        ctx = self._repo.get_short_question_grading_context(attempt_id)
        if ctx is None:
            logger.warning(
                "hydrate_job: no short-question grading context for attempt %s",
                attempt_id,
            )
            return None
        if not isinstance(engram.content, QuizContent):
            logger.warning(
                "hydrate_job: engram %s content is not QuizContent",
                ctx["engram_id"],
            )
            return None

        responses = self._repo.get_short_question_responses(attempt_id)
        if not responses:
            logger.warning(
                "hydrate_job: short-question attempt %s has no saved responses",
                attempt_id,
            )
            return None

        answers = {r.question_index: r.raw_answer for r in responses}
        return ShortQuestionGradingJobPayload(
            job_id=job_id,
            attempt_id=attempt_id,
            engram_id=ctx["engram_id"],
            user_id=ctx["user_id"],
            topic=ctx.get("topic") or "",
            content=engram.content,
            answers=answers,
            target_cognitive_level=ctx["target_cognitive_level"],
            bubble_id=engram.bubble_id or "",
            note_ids=[ctx["note_id"]] if ctx.get("note_id") else [],
        )

    def mark_processing(self, job_id: str) -> None:
        self._repo.update_grading_job(job_id, "processing")

    def mark_done(self, job_id: str) -> None:
        self._repo.update_grading_job(job_id, "done")

    def mark_failed(self, job_id: str, error: str) -> None:
        self._repo.update_grading_job(job_id, "failed", error)


# ---------------------------------------------------------------------------
# Process a single job
# ---------------------------------------------------------------------------


async def process_grading_job(
    repo: MasteryRepository,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    job: GradingJobPayload,
    api_key: Optional[str] = None,
    use_cloud: Optional[bool] = None,
    # api_key falls back to ConfigManager's ollama.api_key when None (see
    # ai_grading.call_grading_model). use_cloud is an explicit per-call
    # override; when None it defers to cfg.ollama.prefer_cloud, matching how
    # engram_generator_inator.py resolves embedding_model / chat_model.
) -> None:
    logger.info("Processing job %s for attempt %s", job.job_id, job.attempt_id)

    # 1. RAG: relevant note chunks — cached retrieval first, main knowledge
    #    base as fallback when the cache is insufficient (see
    #    context_retrieval.retrieve_grading_context). Synchronous FAISS work,
    #    so run it off the event loop.
    note_chunks = await asyncio.to_thread(
        retrieve_grading_context,
        query=job.raw_answer + " " + job.question.question_stem,
        bubble_id=job.bubble_id,
        note_id=job.note_ids[0] if job.note_ids else None,
        topic=job.topic,
        top_k=6,
    )

    # 2. User's past answers for regression context. This uses the (still
    #    abstract) answer vector store, so it's skipped when the worker is run
    #    without a concrete one — grading proceeds without regression context
    #    rather than failing the job.
    past_answers: list[str] = []
    if vector_store is not None and embedder is not None:
        past_records = await retrieve_past_answers(
            vector_store,
            embedder,
            current_answer=job.raw_answer,
            engram_id=job.engram_id,
            user_id=job.user_id,
            limit=3,
        )
        past_answers = [
            f"[Score: {int(r.score * 100)}% | {r.attempted_at[:10]}]\n{r.text}"
            for r in past_records
        ]

    # 3. Current mastery for context
    mastery = repo.get_mastery(job.engram_id, job.user_id)

    # 4. Run grading
    output = await run_grading_pipeline(
        GradingPipelineInput(
            attempt_id=job.attempt_id,
            engram_id=job.engram_id,
            user_id=job.user_id,
            question=job.question,
            user_answer=job.raw_answer,
            target_cognitive_level=job.target_cognitive_level,
            mastery=mastery,
            note_chunks=note_chunks,
            past_answers=past_answers,
            api_key=api_key,
            use_cloud=use_cloud,
        )
    )

    # 5. Index answer embedding (for future regression/similarity lookups).
    #    Also gated on a concrete answer vector store being supplied.
    from datetime import datetime

    vector_id = None
    if vector_store is not None and embedder is not None:
        vector_id = await index_answer(
            vector_store,
            embedder,
            attempt_id=job.attempt_id,
            engram_id=job.engram_id,
            user_id=job.user_id,
            topic=job.topic,
            answer=job.raw_answer,
            score=output.result.score,
            target_cognitive_level=job.target_cognitive_level,
            attempted_at=datetime.utcnow().isoformat(),
        )

    # 6. Apply result → updates mastery, misconceptions, generation queue
    apply_grading_result(
        repo,
        attempt_id=job.attempt_id,
        engram_id=job.engram_id,
        user_id=job.user_id,
        target_cognitive_level=job.target_cognitive_level,
        topic=job.topic,
        result=output.result,
        raw_answer=job.raw_answer,
        vector_id=vector_id,
    )

    logger.info(
        "Job %s done — score %.0f%% | level demonstrated: %d",
        job.job_id,
        output.result.score * 100,
        output.result.level_demonstrated,
    )


async def process_short_question_grading_job(
    repo: MasteryRepository,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    job: ShortQuestionGradingJobPayload,
    api_key: Optional[str] = None,
    use_cloud: Optional[bool] = None,
) -> None:
    logger.info(
        "Processing short-question job %s for attempt %s",
        job.job_id,
        job.attempt_id,
    )

    # RAG: pull note context using the answers + question stems as the query.
    # Cached retrieval first, main knowledge base as fallback (see
    # context_retrieval.retrieve_grading_context). Off-loop for the FAISS work.
    query = " ".join(
        list(job.answers.values())
        + [q.stem for q in job.content.questions]
    )
    note_chunks = await asyncio.to_thread(
        retrieve_grading_context,
        query=query,
        bubble_id=job.bubble_id,
        note_id=job.note_ids[0] if job.note_ids else None,
        topic=job.topic,
        top_k=6,
    )

    mastery = repo.get_mastery(job.engram_id, job.user_id)

    output = await run_short_question_grading_pipeline(
        ShortQuestionGradingPipelineInput(
            attempt_id=job.attempt_id,
            engram_id=job.engram_id,
            user_id=job.user_id,
            content=job.content,
            answers=job.answers,
            target_cognitive_level=job.target_cognitive_level,
            mastery=mastery,
            note_chunks=note_chunks,
            api_key=api_key,
            use_cloud=use_cloud,
        )
    )

    # Unlike long questions, short-answer responses aren't indexed into the
    # vector store for regression tracking — there's no single essay-length
    # answer to embed, and per-sub-answer regression isn't modelled.
    apply_short_question_grading_result(
        repo,
        attempt_id=job.attempt_id,
        engram_id=job.engram_id,
        user_id=job.user_id,
        target_cognitive_level=job.target_cognitive_level,
        topic=job.topic,
        result=output.result,
    )

    logger.info(
        "Short-question job %s done — score %.0f%% across %d question(s)",
        job.job_id,
        output.result.overall_score * 100,
        len(output.result.grades),
    )


# ---------------------------------------------------------------------------
# Process a batch of pending jobs
# ---------------------------------------------------------------------------


async def run_worker_batch(
    repo: MasteryRepository,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    loop: WorkerLoop,
    api_key: Optional[str] = None,
    use_cloud: Optional[bool] = None,
) -> int:
    pending = loop.fetch_pending_jobs(BATCH_SIZE)
    processed = 0

    for job_row in pending:
        job_id = job_row["job_id"]
        attempts = job_row.get("attempts", 0)

        if attempts >= MAX_RETRIES:
            loop.mark_failed(job_id, f"Max retries ({MAX_RETRIES}) exceeded")
            continue

        loop.mark_processing(job_id)
        payload = loop.hydrate_job(job_id, job_row["attempt_id"])

        if not payload:
            loop.mark_failed(job_id, "Could not hydrate job payload")
            continue

        try:
            if isinstance(payload, ShortQuestionGradingJobPayload):
                await process_short_question_grading_job(
                    repo, vector_store, embedder, payload, api_key, use_cloud
                )
            else:
                await process_grading_job(
                    repo, vector_store, embedder, payload, api_key, use_cloud
                )
            loop.mark_done(job_id)
            processed += 1
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            loop.mark_failed(job_id, str(exc))

    return processed


# ---------------------------------------------------------------------------
# Continuous worker loop
# ---------------------------------------------------------------------------


async def run_worker(
    repo: MasteryRepository,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    loop: WorkerLoop,
    api_key: Optional[str] = None,
    use_cloud: Optional[bool] = None,
    poll_interval: int = POLL_INTERVAL,
) -> None:
    logger.info("Grading worker started (polling every %ds)", poll_interval)
    while True:
        n = await run_worker_batch(
            repo, vector_store, embedder, loop, api_key, use_cloud
        )
        if n:
            logger.info("Processed %d job(s)", n)
        await asyncio.sleep(poll_interval)
