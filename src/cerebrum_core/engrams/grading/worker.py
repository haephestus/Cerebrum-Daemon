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

from ..core.mastery_service import MasteryRepository, apply_grading_result
from ..core.types import CognitiveLevel, EngramType, LongQuestionContent
from ..storage.vector_store import (
    EmbeddingProvider,
    VectorStore,
    index_answer,
    retrieve_note_chunks,
    retrieve_past_answers,
)
from .ai_grading import GradingPipelineInput, run_grading_pipeline

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
    note_ids: list[str] = field(default_factory=list)


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
    ) -> Optional[GradingJobPayload]: ...
    def mark_processing(self, job_id: str) -> None: ...
    def mark_done(self, job_id: str) -> None: ...
    def mark_failed(self, job_id: str, error: str) -> None: ...


class SQLiteWorkerLoop:
    """WorkerLoop implementation against MasteryRepository (NoteEngramRepository).

    Only long_question attempts ever produce a grading_jobs row (see
    mastery_service.submit_long_question), so hydrate_job treats a
    non-long_question or missing engram as a hard failure for that job
    rather than silently skipping it.
    """

    def __init__(self, repo: MasteryRepository) -> None:
        self._repo = repo

    def fetch_pending_jobs(self, limit: int) -> list[dict]:
        return self._repo.fetch_pending_grading_jobs(limit=limit)

    def hydrate_job(self, job_id: str, attempt_id: str) -> Optional[GradingJobPayload]:
        ctx = self._repo.get_grading_context(attempt_id)
        if ctx is None:
            logger.warning(
                "hydrate_job: no engram_attempts row for attempt %s", attempt_id
            )
            return None

        engram = self._repo.get_engram(ctx["engram_id"])
        if engram is None:
            logger.warning(
                "hydrate_job: engram %s (attempt %s) not found",
                ctx["engram_id"],
                attempt_id,
            )
            return None
        if engram.type != EngramType.LONG_QUESTION:
            logger.warning(
                "hydrate_job: engram %s is type=%s, not long_question — "
                "grading_jobs should only ever be created for long_question "
                "attempts (see mastery_service.submit_long_question)",
                ctx["engram_id"],
                engram.type,
            )
            return None
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

    # 1. RAG: relevant note chunks
    note_chunks = await retrieve_note_chunks(
        vector_store,
        embedder,
        query=job.raw_answer + " " + job.question.question_stem,
        topic=job.topic,
        top_k=6,
    )

    # 2. User's past answers for regression context
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

    # 5. Index answer embedding
    from datetime import datetime

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
