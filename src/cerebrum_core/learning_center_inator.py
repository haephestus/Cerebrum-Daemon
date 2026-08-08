import asyncio
import json
import logging
from typing import TYPE_CHECKING, Dict, Optional

from agents.rose import RosePrompts
from cerebrum_core.constants import (
    FLASHCARD_SCHEMA,
    LONG_QUESTION_SCHEMA,
    MCQ_SCHEMA,
    SHORT_QUESTION_SCHEMA,
)
from database.note_chunk_registry_inator import NoteChunkRegisterInator
from cerebrum_core.engrams.core import mastery_service
from cerebrum_core.engrams.core.types import EngramType, FlashcardRating
from cerebrum_core.engrams.engram_generator_inator import EngramGenerator
from cerebrum_core.engrams.scheduler.scheduler import build_study_queue
from models.model_inator import NoteStorage
from notes.chunk_analyser_inator import ChunkAnalyserInator
from common.file_util_inator import CerebrumPaths
from notes.note_util_inator import _load_note

if TYPE_CHECKING:
    from database.note_engram_repository import NoteEngramRepository


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Present engram (active grading): pick the next due engram of a given type
# for a student, and record + grade their response when they answer it.
#
# "Due" is decided by scheduler.build_study_queue (lapsed > overdue > due >
# new), which operates on EngramMastery rows — so an engram type filter has
# to be applied by looking up each queued candidate's Engram, since mastery
# rows carry no type of their own (type lives on the shared engram row;
# mastery is per (engram, user)).
# ---------------------------------------------------------------------------


def _fetch_next_engram(
    repo: "mastery_service.MasteryRepository",
    user_id: str,
    engram_type: EngramType,
    topic: Optional[str] = None,
) -> Optional[dict]:
    """Returns the highest-priority due engram of `engram_type` for this
    student, or None if nothing of that type is currently due.

    Return shape: {"engram": Engram, "mastery": EngramMastery,
    "reason": str, "priority": float} — reason/priority pass through
    QueuedEngram so callers/UI can distinguish e.g. "lapsed" from "new".
    """
    masteries = repo.get_all_due_masteries(user_id)
    if topic is not None:
        topic_engram_ids = {e.id for e in repo.get_topic_engrams(user_id, topic)}
        masteries = [m for m in masteries if m.engram_id in topic_engram_ids]

    for queued in build_study_queue(masteries):
        engram = repo.get_engram(queued.engram_id)
        if engram is not None and engram.type == engram_type:
            return {
                "engram": engram,
                "mastery": queued.mastery,
                "reason": queued.reason,
                "priority": queued.priority,
            }
    return None


def fetch_flashcards(
    repo: "mastery_service.MasteryRepository",
    user_id: str,
    topic: Optional[str] = None,
) -> Optional[dict]:
    return _fetch_next_engram(repo, user_id, EngramType.FLASHCARD, topic)


def fetch_short_question(
    repo: "mastery_service.MasteryRepository",
    user_id: str,
    topic: Optional[str] = None,
) -> Optional[dict]:
    return _fetch_next_engram(repo, user_id, EngramType.SHORT_QUESTION, topic)


def fetch_long_questions(
    repo: "mastery_service.MasteryRepository",
    user_id: str,
    topic: Optional[str] = None,
) -> Optional[dict]:
    return _fetch_next_engram(repo, user_id, EngramType.LONG_QUESTION, topic)


def fetch_mcq(
    repo: "mastery_service.MasteryRepository",
    user_id: str,
    topic: Optional[str] = None,
) -> Optional[dict]:
    return _fetch_next_engram(repo, user_id, EngramType.MCQ, topic)


def _topic_for_engram(
    repo: "mastery_service.MasteryRepository", engram_id: str
) -> Optional[str]:
    engram = repo.get_engram(engram_id)
    if not engram:
        return None
    note = repo.get_note(engram.note_id)
    return note.get("topic") if note else None


# ---------------------------------------------------------------------------
# Submit a student's response: thin wrappers around mastery_service's
# process_*_attempt functions, which already handle scoring, scheduling,
# and (on promotion/misconception) queueing further generation. These add
# only a topic-mastery recompute on top, so a student's dashboard reflects
# the attempt immediately rather than waiting for whatever next triggers
# recompute_topic_mastery.
#
# submit_long_question is deliberately NOT auto-recomputed here: it's graded
# asynchronously by the worker, and mastery_service.apply_grading_result
# already calls recompute_topic_mastery once the AI grade actually lands.
# ---------------------------------------------------------------------------


def submit_mcq_answer(
    repo: "mastery_service.MasteryRepository",
    *,
    engram_id: str,
    user_id: str,
    selected_option: str,
    correct_option: str,
    target_cognitive_level: int,
    time_spent_ms: Optional[int] = None,
    topic: Optional[str] = None,
):
    attempt, mastery = mastery_service.process_mcq_attempt(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        selected_option=selected_option,
        correct_option=correct_option,
        target_cognitive_level=target_cognitive_level,
        time_spent_ms=time_spent_ms,
    )
    mastery_service.recompute_topic_mastery(
        repo, user_id, topic or _topic_for_engram(repo, engram_id) or ""
    )
    return attempt, mastery


def submit_flashcard_rating(
    repo: "mastery_service.MasteryRepository",
    *,
    engram_id: str,
    user_id: str,
    rating: FlashcardRating,
    target_cognitive_level: int,
    time_to_flip_ms: Optional[int] = None,
    topic: Optional[str] = None,
):
    attempt, mastery = mastery_service.process_flashcard_attempt(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        rating=rating,
        target_cognitive_level=target_cognitive_level,
        time_to_flip_ms=time_to_flip_ms,
    )
    mastery_service.recompute_topic_mastery(
        repo, user_id, topic or _topic_for_engram(repo, engram_id) or ""
    )
    return attempt, mastery


def submit_short_question_answers(
    repo: "mastery_service.MasteryRepository",
    *,
    engram_id: str,
    user_id: str,
    responses: list[dict],
    target_cognitive_level: int,
    time_spent_ms: Optional[int] = None,
) -> tuple[str, str]:
    """Queues async AI grading (see grading/worker.py) — no score or mastery
    update happens synchronously here. Returns (attempt_id, job_id).

    Short questions are open-response, so (like long questions) they can't be
    scored inline; topic-mastery recompute is deferred to the grading worker's
    apply_short_question_grading_result, same as submit_long_question_answer.
    `responses` is a list of {"question_index": int, "raw_answer": str}.
    """
    return mastery_service.submit_short_question(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        responses=responses,
        target_cognitive_level=target_cognitive_level,
        time_spent_ms=time_spent_ms,
    )


def submit_long_question_answer(
    repo: "mastery_service.MasteryRepository",
    *,
    engram_id: str,
    user_id: str,
    raw_answer: str,
    target_cognitive_level: int,
    time_spent_ms: Optional[int] = None,
    note_version: Optional[int] = None,
    context_snapshot: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Queues async AI grading (see grading/worker.py) — no score or
    mastery update happens synchronously here. Returns (attempt_id, job_id).
    """
    return mastery_service.submit_long_question(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        raw_answer=raw_answer,
        target_cognitive_level=target_cognitive_level,
        time_spent_ms=time_spent_ms,
        note_version=note_version,
        context_snapshot=context_snapshot,
    )


def generate_engram_for_level(
    bubble_id: str,
    note_id: str,
    engram_type: str,
    target_cognitive_level: int,
) -> list[Dict]:
    """
    Mastery-triggered engram generation.

    Runs the same two-pass EngramGenerator flow as the base generators
    (_mcq_generator, _short_question_generator, etc.) but injects a level-aware
    prompt suffix so the output demands deeper cognitive engagement.

    All content-targeting (gaps, confused concepts, weak areas, student
    claims) is pulled automatically from the note's analysis JSON by
    EngramGenerator/_analysis_retriever — nothing about what to focus on
    is passed in here.

    Args:
        bubble_id:        Bubble the note belongs to.
        note_id:          Note to generate from.
        engram_type:      One of 'mcq' | 'flashcard' | 'short_question' | 'long_question'.
        target_cognitive_level:  Target Bloom level (1–7). Drives the prompt suffix.
    """
    schema_map = {
        "mcq": (MCQ_SCHEMA, "rose_mcq_generator"),
        "flashcard": (FLASHCARD_SCHEMA, "rose_flashcard_generator"),
        "short_question": (SHORT_QUESTION_SCHEMA, "rose_short_question_generator"),
        "long_question": (LONG_QUESTION_SCHEMA, "rose_long_question_generator"),
    }

    if engram_type not in schema_map:
        raise ValueError(
            f"Unknown engram_type '{engram_type}'. "
            f"Must be one of: {list(schema_map)}"
        )

    schema, prompt_key = schema_map[engram_type]
    base_prompt = RosePrompts.get_prompt(prompt_key)
    if base_prompt is None:
        raise RuntimeError(f"Prompt '{prompt_key}' not found in RosePrompts")

    level_suffix = LEVEL_SUFFIX.get(target_cognitive_level, "")
    augmented_prompt = base_prompt + level_suffix

    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    engram.target_cognitive_level = target_cognitive_level
    schema_id = schema["schema_id"]

    logger.info(
        "generate_engram_for_level: note=%s type=%s level=%d",
        note_id,
        engram_type,
        target_cognitive_level,
    )

    # Pass 1 — embedding model: RAG retrieval, caches results
    engram.retrieval_pass(
        engram_prompt=augmented_prompt,
        schema_id=schema_id,
    )
    # Pass 2 — chat model: reads from cache, produces long_question output
    outcomes = engram.generation_pass(
        schema_id=schema_id,
        engram_schema=schema,
    )
    return outcomes


def _run_chunk_analysis(bubble_id: str, note: NoteStorage, prompt: str) -> dict:
    """
    Canonical analysis engine — every analysis entrypoint (active or passive)
    funnels through this. There is no whole-note analysis path anymore;
    chunk-based is the only mode.
    """
    registry = NoteChunkRegisterInator()
    note_chunks = registry.fetch_chunks_inator(note.note_id)
    if not note_chunks:
        logger.warning(f"No chunks in registry for note {note.note_id}")
        return {"error": "Note has no registered chunks — run chunking pipeline first"}

    if note.analyse_note is False:
        return {}

    analyser = ChunkAnalyserInator(
        bubble_id=bubble_id,
        note_id=note.note_id,
        note_chunks=note_chunks,
        note=note,
    )

    chunk_analyses: dict[int, dict] = {}
    errors: list[str] = []
    for result in analyser.chunk_stream_inator(prompt=prompt, top_k_chunks=5):
        if result.status == "error":
            logger.warning(f"Chunk {result.chunk_index} failed: {result.error}")
            errors.append(f"chunk_{result.chunk_index}: {result.error}")
        else:
            logger.info(f"Chunk {result.chunk_index} — {result.status}")
            chunk_analyses[result.chunk_index] = result.analysis

    if not chunk_analyses:
        logger.warning("No chunks analysed successfully")
        return {"error": "No chunks could be analysed", "details": errors}

    result = {
        "chunk_diagnostics": [
            finding
            for idx in sorted(chunk_analyses)
            for finding in chunk_analyses[idx].get("chunk_diagnostics", [])
        ],
        "note_overview": chunk_analyses[max(chunk_analyses)].get("note_overview", {}),
        "metadata": {
            "note_id": note.note_id,
            "bubble_id": bubble_id,
            "content_version": note.metadata.content_version,
            "note_title": getattr(note.metadata, "title", ""),
            "chunks_count": len(note_chunks),
            "errors": errors,
        },
    }

    # Completion: this is where analysis results actually land in the system.
    # The current analysis is already up to date as version-keyed JSON (the
    # chunk cache written during chunk_stream_inator). Now (1) assign the
    # note's topic from the overview so mastery/engrams/planner can group on
    # it, and (2) persist this completed analysis into the queryable
    # vectorstore analysis cache for overview development. Best-effort: a
    # failure here is logged but doesn't discard the analysis result.
    _finalise_analysis(bubble_id, note, result)
    return result


def _finalise_analysis(bubble_id: str, note: NoteStorage, result: dict) -> None:
    """Persist side-effects of a completed analysis: topic assignment on the
    note (topic entity) and historical persistence of the overview/findings
    into the vectorstore analysis cache."""
    overview = result.get("note_overview") or {}
    findings = result.get("chunk_diagnostics") or []

    # (1) Assign the topic to the note via the topic entity.
    topic = overview.get("topic")
    if topic:
        try:
            from database.note_engram_repository import (
                NoteEngramRepository,
            )

            resolved = NoteEngramRepository().assign_note_topic(note.note_id, topic)
            if resolved:
                logger.info(
                    "Assigned topic %r (id=%s) to note %s",
                    resolved["name"],
                    resolved["id"],
                    note.note_id,
                )
            else:
                logger.warning(
                    "Topic assignment skipped for note %s (note not registered?)",
                    note.note_id,
                )
        except Exception:
            logger.exception("Failed to assign topic for note %s", note.note_id)

    # (2) Historically persist the analysis into the vectorstore analysis cache.
    try:
        from common.cache_inator import AnalysisVectorCacheInator

        AnalysisVectorCacheInator(note.note_id, bubble_id).persist(
            note.metadata.content_version, overview, findings
        )
    except Exception:
        logger.exception(
            "Failed to persist analysis history for note %s", note.note_id
        )


def _update_profile_from_note_analysis(bubble_id: str, note) -> None:
    """Best-effort write-time learning-profile inference from a fresh note
    analysis. Resolves the note's owner (NoteStorage.user_id) and folds the
    analysis overview's structural signals (e.g. confused_links) into their
    inferred profile. Isolated + swallowed so it can never break analysis."""
    try:
        from common.cache_inator import AnalysisCacheInator
        from database.note_engram_repository import NoteEngramRepository

        from cerebrum_core import learning_profile_inference_inator as lpi

        repo = NoteEngramRepository()
        # Owner lives in the notes table, not on NoteStorage.
        user_id = (repo.get_note(note.note_id) or {}).get("user_id")
        if not user_id:
            return
        overview = AnalysisCacheInator(
            bubble_id=bubble_id, note_id=note.note_id
        ).get_cached_overview(note.metadata.content_version)
        if not overview:
            return
        lpi.apply_note_analysis(
            repo, user_id, {"note_overview": overview}, note_id=note.note_id
        )
    except Exception:
        logger.debug("learning-profile note-analysis update skipped", exc_info=True)


def _precompute_kb_suggestions(bubble_id: str, note) -> None:
    """Best-effort: after analysis, precompute KB-first (offline) suggested
    readings for the note so they're ready when the client asks. External
    providers stay on-demand (they cost network); this only touches the KB.
    Never raises into the analysis flow."""
    try:
        from common.cache_inator import AnalysisCacheInator
        from database.file_registry_inator import FileRegisterInator
        from database.note_engram_repository import NoteEngramRepository

        from cerebrum_core.knowledgebase_inator import KnowledgebaseManager
        from cerebrum_core.suggested_reading_inator import (
            build_seed_from_overview,
            suggest,
        )
        from cerebrum_core.user_context_inator import build_effective_profile

        repo = NoteEngramRepository()
        # Owner lives in the notes table, not on NoteStorage.
        user_id = (repo.get_note(note.note_id) or {}).get("user_id")
        if not user_id:
            return
        overview = AnalysisCacheInator(
            bubble_id=bubble_id, note_id=note.note_id
        ).get_cached_overview(note.metadata.content_version)
        if not overview:
            return
        suggest(
            repo,
            seed=build_seed_from_overview(overview),
            seed_ref=note.note_id,
            user_id=user_id,
            manager=KnowledgebaseManager(),
            file_registry=FileRegisterInator(),
            org_ids=repo.get_user_org_ids(user_id),
            include_external=False,  # offline-safe; external is on-demand
            effective_profile=build_effective_profile(repo, user_id),
        )
    except Exception:
        logger.debug("kb suggested-reading precompute skipped", exc_info=True)


def active_analysis(bubble_id: str, filename: str) -> dict:
    # Was: reading `CerebrumPaths().note_path(...)` directly with
    # `.read_text()` — broke once notes moved to folder-form storage
    # (content.json + ink.json), since that path stopped existing for
    # any migrated note. `_load_note` is the shared, storage-shape-aware
    # loader used everywhere else (see note_util_inator.py) — this now
    # goes through the same path as bubble_router.py's endpoints instead
    # of re-deriving note storage layout itself.
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    note = _load_note(notes_dir, filename)
    prompt = RosePrompts.get_prompt("rose_note_analyser")
    if not prompt:
        return {"error": "Prompt cannot be none"}
    try:
        logger.info(
            f"Starting active analysis (chunk) for note {note.note_id} "
            f"v{note.metadata.content_version}"
        )
        result = _run_chunk_analysis(bubble_id, note, prompt)
        _update_profile_from_note_analysis(bubble_id, note)
        _precompute_kb_suggestions(bubble_id, note)
        return result
    except Exception as e:
        logger.error(
            f"Failed chunk analysis for note {note.note_id}: {e}", exc_info=True
        )
        raise


def passive_analysis(
    bubble_id: str,
    note: NoteStorage,
    prompt: str,
) -> dict:
    logger.info(
        f"Starting passive analysis for note {note.note_id} v{note.metadata.content_version}"
    )
    result = _run_chunk_analysis(bubble_id, note, prompt)
    if not result:
        return {"result": "no analysis for this note"}
    _update_profile_from_note_analysis(bubble_id, note)
    _precompute_kb_suggestions(bubble_id, note)
    logger.info(f"Completed passive analysis for note {note.note_id}")
    return result


# ---------------------------------------------------------------------------
# 1. MODULE-LEVEL CONSTANT  (paste near the top, after imports)
# ---------------------------------------------------------------------------

# Prompt suffixes appended to the base generator prompts when producing
# mastery-triggered engrams at a specific cognitive level. 7 levels:
# Remember/Understand/Apply/Analyse/Synthesise/Evaluate/Doctoral. All
# levels have explicit generation instructions — none are blank, since a
# blank suffix gives the model no signal about what "level 1" or "level 2"
# actually means beyond the base prompt.
LEVEL_SUFFIX: dict[int, str] = {
    1: (
        "\n\nGenerate questions that test RECALL only. The student must retrieve a "
        "fact, term, definition, or basic concept directly stated in the notes — "
        "define, list, name, identify, or state it verbatim or near-verbatim. "
        "Do not require explanation, application, or comparison. A correct answer "
        "should be gradable by exact or near-exact match to the source material."
    ),
    2: (
        "\n\nGenerate questions that require UNDERSTANDING, not just recall. The "
        "student must explain, describe, paraphrase, classify, or summarise the "
        "concept in their own words, or identify a correct example of it. The "
        "correct answer must demonstrate comprehension of meaning, not just "
        "recitation of the source text — reject answers that only copy the notes' "
        "phrasing without showing the idea has been understood."
    ),
    3: (
        "\n\nGenerate questions that require APPLYING this concept to new scenarios. "
        "The student must demonstrate they can use the knowledge, not just recall it. "
        "Do not ask for definitions."
    ),
    4: (
        "\n\nGenerate questions that require ANALYSING this concept — breaking it down, "
        "explaining why it occurs, identifying edge cases, and evaluating the underlying "
        "mechanism. Surface-level recall is not sufficient for a passing answer."
    ),
    5: (
        "\n\nGenerate questions that require SYNTHESISING this concept with at least one "
        "other concept from the source material. The student must propose or explain a "
        "connection that is not explicitly stated in the notes. "
        "A correct but isolated answer scores poorly."
    ),
    6: (
        "\n\nGenerate questions that require EVALUATING and CRITIQUING. The student must "
        "defend a position with evidence, identify the limitations of a given approach, "
        "and argue against at least one counterpoint. "
        "There is no single correct answer — assess quality of reasoning."
    ),
    7: (
        "\n\nGenerate DOCTORAL-LEVEL questions. Ask about open problems, research gaps, "
        "competing hypotheses, or novel applications not covered in the notes. "
        "The student is expected to reason at the frontier of the topic. "
        "Assess depth of independent thought, not factual recall."
    ),
}

LEVEL_NAME: dict[int, str] = {
    1: "recall",
    2: "understand",
    3: "apply",
    4: "analyse",
    5: "synthesise",
    6: "evaluate",
    7: "doctoral",
}

# ---------------------------------------------------------------------------
# 2. generate_engram_for_level()  (paste alongside _mcq_generator etc.)
# ---------------------------------------------------------------------------


def process_generation_queue(repo: "NoteEngramRepository", limit: int = 10) -> int:
    jobs = repo.fetch_pending_generation_jobs(limit=limit)
    if not jobs:
        return 0

    processed = 0
    for job in jobs:
        job_id = job["id"]
        try:
            note = repo.get_note(job["note_id"])
            if not note:
                repo.mark_generation_job_failed(job_id, "note not found in DB")
                logger.warning(
                    "Generation job %s skipped: note %s not found",
                    job_id,
                    job["note_id"],
                )
                continue

            bubble_id = note.get("bubble_id")
            if not bubble_id:
                repo.mark_generation_job_failed(job_id, "note has no bubble_id")
                logger.warning(
                    "Generation job %s skipped: note %s has no bubble_id",
                    job_id,
                    job["note_id"],
                )
                continue

            outcomes = generate_engram_for_level(
                bubble_id=bubble_id,
                note_id=job["note_id"],
                engram_type=job["target_type"],
                target_cognitive_level=int(job["target_cognitive_level"]),
            )

            succeeded = [o for o in outcomes if o["engram_id"] is not None]
            failed = [o for o in outcomes if o["engram_id"] is None]

            # Seed mastery=NEW rows for this job's target student. Without
            # this, a freshly generated engram is invisible to
            # scheduler.build_study_queue (which iterates EngramMastery
            # rows, not engrams) until the student has already attempted
            # it once — which never happens if they can't see it to
            # attempt it in the first place. mastery_service._update_mastery
            # still does the same lazy-create for the general case (an
            # engram a student reaches some other way); this just also
            # covers the queue-driven path where we already know exactly
            # who the engram is for.
            target_user_id = job.get("user_id")
            if target_user_id:
                seeded_engram_ids: set[str] = set()
                for outcome in succeeded:
                    eid = outcome["engram_id"]
                    if eid in seeded_engram_ids:
                        continue
                    seeded_engram_ids.add(eid)
                    if repo.get_mastery(eid, target_user_id) is None:
                        repo.upsert_mastery(
                            mastery_service.create_initial_mastery(eid, target_user_id)
                        )

            if not outcomes:
                # generation_pass had cache files but produced zero outcomes —
                # shouldn't happen given the loop structure above, but treat
                # defensively as a failure rather than a silent success.
                repo.mark_generation_job_failed(
                    job_id, "generation_pass returned no outcomes"
                )
                logger.error("Generation job %s failed: no outcomes produced", job_id)
                continue

            if not succeeded:
                error_summary = "; ".join(
                    o["error"] or "unknown error" for o in failed[:5]
                )
                repo.mark_generation_job_failed(
                    job_id,
                    f"all {len(failed)} item(s) failed: {error_summary}",
                )
                logger.error(
                    "Generation job %s failed: 0/%d items produced an engram",
                    job_id,
                    len(outcomes),
                )
                continue

            if failed:
                # Partial success: some engrams landed, some didn't. Marked
                # done since the job did produce usable content, but the
                # failures are logged loudly rather than swallowed.
                logger.warning(
                    "Generation job %s partially failed: %d/%d items succeeded",
                    job_id,
                    len(succeeded),
                    len(outcomes),
                )

            repo.mark_generation_job_done(job_id)
            processed += 1

            logger.info(
                "Generation job %s done: note=%s type=%s level=%d trigger=%s",
                job_id,
                job["note_id"],
                job["target_type"],
                job["target_cognitive_level"],
                job.get("trigger", "unknown"),
            )

        except Exception as exc:
            logger.error(
                "Generation job %s failed: %s",
                job_id,
                exc,
                exc_info=True,
            )
            repo.mark_generation_job_failed(job_id, str(exc))

    return processed


# ---------------------------------------------------------------------------
# Background poll loop for process_generation_queue.
#
# process_generation_queue existed but nothing called it — rows written by
# mastery_service.queue_engram_generation (on misconception detection and
# level promotion) just accumulated. This mirrors grading.worker.run_worker's
# poll shape so the two queues (grading_jobs, engram_generation_queue) are
# drained the same way; run this alongside grading.worker.run_worker in
# whatever process/lifespan hosts the background workers, e.g.:
#
#     asyncio.gather(
#         worker.run_worker(repo, vector_store, embedder, SQLiteWorkerLoop(repo)),
#         run_generation_queue_worker(repo),
#     )
# ---------------------------------------------------------------------------

GENERATION_QUEUE_POLL_INTERVAL = 60  # seconds
GENERATION_QUEUE_BATCH_SIZE = 10


async def run_generation_queue_worker(
    repo: "NoteEngramRepository",
    poll_interval: int = GENERATION_QUEUE_POLL_INTERVAL,
    limit: int = GENERATION_QUEUE_BATCH_SIZE,
) -> None:
    """Continuously drains engram_generation_queue via process_generation_queue.

    process_generation_queue() is synchronous end-to-end (file I/O in
    EngramGenerator plus blocking ollama_cloud_call/ollama_local_call calls),
    so each poll is pushed onto a worker thread rather than run inline —
    otherwise it would block this event loop for the duration of every
    generation job, same reasoning as ai_grading.call_grading_model's
    asyncio.to_thread wrapping.
    """
    logger.info(
        "Engram generation queue worker started (polling every %ds)", poll_interval
    )
    while True:
        try:
            n = await asyncio.to_thread(process_generation_queue, repo, limit)
            if n:
                logger.info("Generation queue: processed %d job(s)", n)
        except Exception:
            logger.exception("Generation queue poll failed")
        await asyncio.sleep(poll_interval)
