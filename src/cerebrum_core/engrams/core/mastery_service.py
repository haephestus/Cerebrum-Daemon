"""
cerebrum_core.engrams.core.mastery_service
============================================
Orchestrates: attempt recording → grading → scheduling → topic rollup.
Defines the MasteryRepository ABC and all process_* entry points.

NOTE: this reverts an earlier revision that renamed everything from `topic`
to `domain`. That rename was a mistake: `domain` is the pipeline-tracking
classification set on `notes` by mark_analysed_inator, while `topic` is what
the note is about — the field engrams/mastery actually group and roll up on.
TopicMastery in types.py only ever had a `.topic` attribute, never `.domain`.
Everything below is back to topic.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ..scheduler.scheduler import apply_schedule, compute_schedule
from .scoring import flashcard_rating_to_score, score_mcq, score_short_question
from .types import (
    DimensionScores,
    Engram,
    EngramAttempt,
    EngramMastery,
    EngramType,
    FlashcardRating,
    FlashcardResponse,
    GraderType,
    GradingResult,
    LongQuestionResponse,
    MasteryState,
    MasteryVector,
    MCQResponse,
    QuizResponse,
    ShortQuestionGradingResult,
    TopicMastery,
)


def _now() -> str:
    return datetime.utcnow().isoformat()


def _nanoid() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Repository interface (implement with SQLite, Postgres, etc.)
# ---------------------------------------------------------------------------


class MasteryRepository(ABC):

    # Attempts
    @abstractmethod
    def create_attempt(self, attempt: EngramAttempt) -> None: ...

    @abstractmethod
    def update_attempt_score(
        self, attempt_id: str, score: float, grader: GraderType
    ) -> None: ...

    @abstractmethod
    def get_recent_attempt_scores(
        self, engram_id: str, user_id: str, limit: int = 10
    ) -> list[float]: ...

    # Type-specific responses
    @abstractmethod
    def save_mcq_response(self, r: MCQResponse) -> None: ...

    @abstractmethod
    def save_flashcard_response(self, r: FlashcardResponse) -> None: ...

    @abstractmethod
    def save_short_question_responses(self, responses: list[QuizResponse]) -> None: ...

    @abstractmethod
    def get_short_question_responses(self, attempt_id: str) -> list[QuizResponse]: ...

    @abstractmethod
    def save_short_question_grades(self, grades: list[QuizResponse]) -> None: ...

    @abstractmethod
    def save_long_question_response(self, r: LongQuestionResponse) -> None: ...

    @abstractmethod
    def get_long_question_responses(
        self, engram_id: str, user_id: str, limit: int = 5
    ) -> list[LongQuestionResponse]: ...

    # Mastery
    @abstractmethod
    def get_mastery(self, engram_id: str, user_id: str) -> Optional[EngramMastery]: ...

    @abstractmethod
    def upsert_mastery(self, mastery: EngramMastery) -> None: ...

    @abstractmethod
    def get_topic_masteries(self, user_id: str, topic: str) -> list[EngramMastery]: ...

    @abstractmethod
    def get_all_due_masteries(self, user_id: str) -> list[EngramMastery]: ...

    # Topic entity + rollup
    @abstractmethod
    def resolve_topic(self, user_id: str, name: str) -> Optional[dict]: ...

    @abstractmethod
    def upsert_topic_mastery(self, tm: TopicMastery) -> None: ...

    @abstractmethod
    def get_topic_mastery(self, user_id: str, topic: str) -> Optional[TopicMastery]: ...

    # Misconceptions
    @abstractmethod
    def upsert_misconception(
        self, user_id: str, engram_id: str, concept: str, description: str
    ) -> None: ...

    # Atomic submissions (attempt + responses + grading job in one txn)
    @abstractmethod
    def record_short_question_submission(
        self, attempt: EngramAttempt, responses: list[QuizResponse], priority: int = 5
    ) -> str: ...

    @abstractmethod
    def record_long_question_submission(
        self, attempt: EngramAttempt, response: LongQuestionResponse, priority: int = 5
    ) -> str: ...

    # Grading jobs
    @abstractmethod
    def create_grading_job(self, attempt_id: str, priority: int = 5) -> str: ...

    @abstractmethod
    def update_grading_job(
        self, job_id: str, status: str, error: Optional[str] = None
    ) -> None: ...

    # Grading-worker consumer (backs grading.worker.SQLiteWorkerLoop)
    @abstractmethod
    def fetch_pending_grading_jobs(self, limit: int = 10) -> list[dict]: ...

    @abstractmethod
    def get_grading_context(self, attempt_id: str) -> Optional[dict]: ...

    @abstractmethod
    def get_short_question_grading_context(
        self, attempt_id: str
    ) -> Optional[dict]: ...

    # Engram generation queue
    @abstractmethod
    def queue_engram_generation(
        self,
        bubble_id: str,
        note_id: str,
        user_id: str,
        trigger: str,
        target_cognitive_level: int,
        target_type: str,
        trigger_ref: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> None: ...

    # Generation queue consumer
    @abstractmethod
    def fetch_pending_generation_jobs(self, limit: int = 10) -> list[dict]: ...

    @abstractmethod
    def mark_generation_job_done(self, job_id: str) -> None: ...

    @abstractmethod
    def mark_generation_job_failed(self, job_id: str, error: str) -> None: ...

    # Notes (read-only lookup — note registration/ingestion itself is a
    # NoteEngramRepository-specific concern, but get_note is needed by
    # anything that must resolve an engram back to its topic, e.g.
    # learning_center_inator._topic_for_engram)
    @abstractmethod
    def get_note(self, note_id: str) -> Optional[dict]: ...

    # Engrams
    @abstractmethod
    def create_engram(self, engram: Engram) -> None: ...

    @abstractmethod
    def get_engram(self, engram_id: str) -> Optional[Engram]: ...

    @abstractmethod
    def get_topic_engrams(self, user_id: str, topic: str) -> list[Engram]: ...


# ---------------------------------------------------------------------------
# Initial mastery factory
# ---------------------------------------------------------------------------


def create_initial_mastery(engram_id: str, user_id: str) -> EngramMastery:
    return EngramMastery(
        id=_nanoid(),
        engram_id=engram_id,
        user_id=user_id,
        state=MasteryState.NEW,
        next_due_at=_now(),
    )


# ---------------------------------------------------------------------------
# Process MCQ attempt
# ---------------------------------------------------------------------------


def process_mcq_attempt(
    repo: MasteryRepository,
    *,
    engram_id: str,
    user_id: str,
    selected_option: str,
    correct_option: str,
    target_cognitive_level: int,
    time_spent_ms: Optional[int] = None,
) -> tuple[EngramAttempt, EngramMastery]:
    is_correct = selected_option == correct_option
    score = score_mcq(is_correct)
    attempt_id = _nanoid()

    attempt = EngramAttempt(
        id=attempt_id,
        engram_id=engram_id,
        user_id=user_id,
        target_cognitive_level=target_cognitive_level,
        score=score,
        grader=GraderType.AUTO,
        time_spent_ms=time_spent_ms,
    )
    repo.create_attempt(attempt)
    repo.save_mcq_response(
        MCQResponse(
            attempt_id=attempt_id,
            selected_option=selected_option,
            correct_option=correct_option,
            is_correct=is_correct,
            distractor_key=None if is_correct else selected_option,
        )
    )

    mastery = _update_mastery(repo, engram_id, user_id, score)
    return attempt, mastery


# ---------------------------------------------------------------------------
# Process Flashcard attempt
# ---------------------------------------------------------------------------


def process_flashcard_attempt(
    repo: MasteryRepository,
    *,
    engram_id: str,
    user_id: str,
    rating: FlashcardRating,
    target_cognitive_level: int,
    time_to_flip_ms: Optional[int] = None,
) -> tuple[EngramAttempt, EngramMastery]:
    score = flashcard_rating_to_score(rating)
    attempt_id = _nanoid()

    attempt = EngramAttempt(
        id=attempt_id,
        engram_id=engram_id,
        user_id=user_id,
        target_cognitive_level=target_cognitive_level,
        score=score,
        grader=GraderType.AUTO,
    )
    repo.create_attempt(attempt)
    repo.save_flashcard_response(
        FlashcardResponse(
            attempt_id=attempt_id,
            self_rating=rating,
            time_to_flip_ms=time_to_flip_ms,
        )
    )

    mastery = _update_mastery(repo, engram_id, user_id, score)
    return attempt, mastery


# ---------------------------------------------------------------------------
# Submit short-question attempt (queues async grading job)
#
# Short questions are open-response, so — like long questions — they can't be
# scored synchronously here. This records the attempt as pending, stores the
# raw sub-answers, and enqueues ONE grading_jobs row for the whole attempt.
# The grading worker later grades every sub-answer, writes the grades back,
# and calls apply_short_question_grading_result to update score/mastery.
# ---------------------------------------------------------------------------


def submit_short_question(
    repo: MasteryRepository,
    *,
    engram_id: str,
    user_id: str,
    responses: list[dict],
    target_cognitive_level: int,
    time_spent_ms: Optional[int] = None,
) -> tuple[str, str]:
    """Queues async AI grading. `responses` is a list of
    {"question_index": int, "raw_answer": str}. Returns (attempt_id, job_id).
    No score or mastery update happens synchronously.
    """
    attempt_id = _nanoid()

    attempt = EngramAttempt(
        id=attempt_id,
        engram_id=engram_id,
        user_id=user_id,
        target_cognitive_level=target_cognitive_level,
        score=None,
        grader=GraderType.PENDING,
        time_spent_ms=time_spent_ms,
    )
    response_rows = [
        QuizResponse(
            id=_nanoid(),
            attempt_id=attempt_id,
            question_index=r["question_index"],
            raw_answer=r["raw_answer"],
        )
        for r in responses
    ]
    priority = min(10, 4 + target_cognitive_level)
    # Attempt + responses + grading job land in one transaction.
    job_id = repo.record_short_question_submission(attempt, response_rows, priority)
    return attempt_id, job_id


# ---------------------------------------------------------------------------
# Apply short-question grading result (called by the worker once the AI grade
# lands). Mirrors apply_grading_result for long questions: write grades back,
# score the attempt, roll misconceptions + mastery + topic forward.
# ---------------------------------------------------------------------------


def apply_short_question_grading_result(
    repo: MasteryRepository,
    *,
    attempt_id: str,
    engram_id: str,
    user_id: str,
    target_cognitive_level: int,
    topic: str,
    result: ShortQuestionGradingResult,
) -> EngramMastery:
    graded_at = _now()

    # 1. Persist per-question grades back onto the response rows.
    repo.save_short_question_grades(
        [
            QuizResponse(
                id="",  # ignored by save_short_question_grades (matches on
                #        attempt_id + question_index, not row id)
                attempt_id=attempt_id,
                question_index=g.question_index,
                raw_answer="",  # not written by the grade UPDATE
                score=g.score,
                is_correct=g.is_correct,
                feedback=g.feedback,
                misconceptions=g.misconceptions,
                graded_at=graded_at,
            )
            for g in result.grades
        ]
    )

    # 2. Score the attempt from the per-question float scores.
    score = score_short_question(
        [{"score": g.score} for g in result.grades]
    )
    repo.update_attempt_score(attempt_id, score, GraderType.AI)

    # 3. Record misconceptions (feeds the misconception dashboard and the
    #    generation trigger below), same as the long-question path.
    for m in result.misconceptions:
        if not m.get("concept") or not m.get("description"):
            continue
        repo.upsert_misconception(
            user_id=user_id,
            engram_id=engram_id,
            concept=m["concept"],
            description=m["description"],
        )

    # 4. Update mastery/scheduling.
    mastery = _update_mastery(repo, engram_id, user_id, score)

    # 5. Queue targeted regeneration on misconceptions, mirroring
    #    apply_grading_result. Target type is chosen by _type_for_level for
    #    the suggested next level rather than hardcoding long_question.
    if result.misconceptions:
        engram = repo.get_engram(engram_id)
        if engram:
            repo.queue_engram_generation(
                bubble_id=engram.bubble_id,
                note_id=engram.note_id,
                user_id=user_id,
                trigger="misconception",
                trigger_ref=attempt_id,
                target_cognitive_level=result.suggested_next_level,
                target_type=_type_for_level(result.suggested_next_level),
                instructions=json.dumps(
                    {
                        "focus_concepts": result.concepts_missed,
                        "misconceptions": result.misconceptions,
                    }
                ),
            )

    recompute_topic_mastery(repo, user_id, topic)
    return mastery


# ---------------------------------------------------------------------------
# Submit Long Question (queues async grading job)
# ---------------------------------------------------------------------------


def submit_long_question(
    repo: MasteryRepository,
    *,
    engram_id: str,
    user_id: str,
    raw_answer: str,
    target_cognitive_level: int,
    time_spent_ms: Optional[int] = None,
    note_version: Optional[int] = None,
    context_snapshot: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Returns (attempt_id, job_id). job_id is the real grading_jobs row id
    returned by create_grading_job, so the caller can poll it."""
    attempt_id = _nanoid()

    attempt = EngramAttempt(
        id=attempt_id,
        engram_id=engram_id,
        user_id=user_id,
        target_cognitive_level=target_cognitive_level,
        score=None,
        grader=GraderType.PENDING,
        time_spent_ms=time_spent_ms,
        note_version=note_version,
        context_snapshot=context_snapshot,
    )
    response = LongQuestionResponse(
        attempt_id=attempt_id,
        raw_answer=raw_answer,
        word_count=len(raw_answer.split()),
    )
    # TODO: add raw_answer to vector store
    priority = min(10, 4 + target_cognitive_level)
    # Attempt + response + grading job land in one transaction.
    job_id = repo.record_long_question_submission(attempt, response, priority)

    return attempt_id, job_id


# ---------------------------------------------------------------------------
# Apply AI grading result (called by worker after grading job completes)
# ---------------------------------------------------------------------------


def apply_grading_result(
    repo: MasteryRepository,
    *,
    attempt_id: str,
    engram_id: str,
    user_id: str,
    target_cognitive_level: int,
    topic: str,
    result: GradingResult,
    raw_answer: str,
    vector_id: Optional[str] = None,
) -> EngramMastery:
    repo.update_attempt_score(attempt_id, result.score, GraderType.AI)
    repo.save_long_question_response(
        LongQuestionResponse(
            attempt_id=attempt_id,
            raw_answer=raw_answer,
            word_count=len(raw_answer.split()),
            ai_feedback=result.feedback,
            concepts_demonstrated=result.concepts_demonstrated,
            concepts_missed=result.concepts_missed,
            misconceptions=result.misconceptions,
            dimension_scores=result.dimension_scores,
            level_demonstrated=result.level_demonstrated,
            regression_detected=result.regression_from_last,
            vector_id=vector_id,
            graded_at=_now(),
        )
    )

    for m in result.misconceptions:
        repo.upsert_misconception(
            user_id=user_id,
            engram_id=engram_id,
            concept=m["concept"],
            description=m["description"],
        )

    mastery = _update_mastery(
        repo, engram_id, user_id, result.score, result.dimension_scores
    )

    # Queue engram generation targeting misconception gaps
    if result.misconceptions:
        engram = repo.get_engram(engram_id)
        if engram:
            repo.queue_engram_generation(
                bubble_id=engram.bubble_id,
                note_id=engram.note_id,
                user_id=user_id,
                trigger="misconception",
                trigger_ref=attempt_id,
                target_cognitive_level=result.suggested_next_level,
                target_type=EngramType.LONG_QUESTION.value,
                instructions=json.dumps(
                    {
                        "focus_concepts": result.concepts_missed,
                        "misconceptions": result.misconceptions,
                    }
                ),
            )

    recompute_topic_mastery(repo, user_id, topic)
    return mastery


# ---------------------------------------------------------------------------
# Internal: update mastery after any attempt
# Checks for level promotion and queues new engram generation when it occurs.
# ---------------------------------------------------------------------------


def _update_mastery(
    repo: MasteryRepository,
    engram_id: str,
    user_id: str,
    score: float,
    dimensions: Optional[DimensionScores] = None,
) -> EngramMastery:
    mastery = repo.get_mastery(engram_id, user_id) or create_initial_mastery(
        engram_id, user_id
    )
    recent_scores = repo.get_recent_attempt_scores(engram_id, user_id, 10)
    decision = compute_schedule(mastery, score, recent_scores, dimensions)
    updated = apply_schedule(mastery, decision, score, dimensions)
    repo.upsert_mastery(updated)

    # When the scheduler promotes to a new cognitive level, queue generation
    # of harder engrams from the same source note at the new level.
    if decision.promotion_occurred:
        engram = repo.get_engram(engram_id)
        if engram:
            repo.queue_engram_generation(
                bubble_id=engram.bubble_id,
                note_id=engram.note_id,
                user_id=user_id,
                trigger="level_promotion",
                trigger_ref=engram_id,
                target_cognitive_level=decision.new_level,
                target_type=_type_for_level(decision.new_level),
            )

    return updated


def _type_for_level(level: int) -> str:
    """
    Which engram type to generate at each cognitive level.

    Levels 1-2: flashcards and MCQ cover recall/comprehension well.
    Levels 3-4: MCQ can test application/analysis with scenario questions;
                short_question open-response is also appropriate here.
    Levels 5+:  only long questions can assess synthesis, evaluation, and
                doctoral-level original thought — MCQ/flashcard can't.
    """
    if level >= 5:
        return EngramType.LONG_QUESTION.value
    if level >= 3:
        return EngramType.SHORT_QUESTION.value
    return EngramType.MCQ.value


# ---------------------------------------------------------------------------
# Topic mastery rollup
# ---------------------------------------------------------------------------


def _update_profile_from_perf(repo, user_id: str) -> None:
    """Best-effort write-time learning-profile inference from topic mastery.

    recompute_topic_mastery is the single sink every mastery-affecting path
    funnels through (mcq/flashcard submit, async short/long grading), so this
    is where an updated engram-performance signal folds into the learner's
    inferred profile. Isolated + swallowed: a profile-inference failure (or a
    MasteryRepository impl without the learning-profile methods, e.g. a test
    stub) must never break a mastery update.
    """
    try:
        from cerebrum_core import learning_profile_inference_inator as lpi

        lpi.apply_engram_performance(repo, user_id)
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "learning-profile perf update skipped", exc_info=True
        )


def recompute_topic_mastery(
    repo: MasteryRepository,
    user_id: str,
    topic: str,
) -> TopicMastery:
    # Resolve the topic to its entity so the rollup row carries a stable
    # topic_id (and a canonical display name), keeping topic_mastery aligned
    # with notes.topic_id. resolved is None only for a blank topic.
    resolved = repo.resolve_topic(user_id, topic)
    topic_id = resolved["id"] if resolved else None
    topic = resolved["name"] if resolved else topic

    masteries = repo.get_topic_masteries(user_id, topic)

    if not masteries:
        tm = TopicMastery(
            id=_nanoid(), user_id=user_id, topic=topic, topic_id=topic_id
        )
        repo.upsert_topic_mastery(tm)
        return tm

    def band_avg(ms: list[EngramMastery]) -> float:
        return sum(m.current_score for m in ms) / len(ms) if ms else 0.0

    factual = [m for m in masteries if m.current_level <= 2]
    applied = [m for m in masteries if 3 <= m.current_level <= 4]
    conceptual = [m for m in masteries if 5 <= m.current_level <= 6]
    doctoral = [m for m in masteries if m.current_level == 7]

    f_score = band_avg(factual)
    a_score = band_avg(applied)
    c_score = band_avg(conceptual)
    d_score = band_avg(doctoral)

    weights = [
        (f_score, 1.0 if factual else 0.0),
        (a_score, 1.5 if applied else 0.0),
        (c_score, 2.0 if conceptual else 0.0),
        (d_score, 2.5 if doctoral else 0.0),
    ]
    total_w = sum(w for _, w in weights)
    overall = sum(s * w for s, w in weights) / total_w if total_w else 0.0

    existing = repo.get_topic_mastery(user_id, topic)
    tm = TopicMastery(
        id=existing.id if existing else _nanoid(),
        user_id=user_id,
        topic=topic,
        topic_id=topic_id,
        factual_score=f_score,
        applied_score=a_score,
        conceptual_score=c_score,
        doctoral_score=d_score,
        overall_score=overall,
        engram_count=len(masteries),
        lapsed_count=sum(1 for m in masteries if m.state == MasteryState.LAPSED),
        updated_at=_now(),
    )
    repo.upsert_topic_mastery(tm)
    _update_profile_from_perf(repo, user_id)
    return tm


def to_mastery_vector(tm: TopicMastery) -> MasteryVector:
    return MasteryVector(
        factual=tm.factual_score,
        applied=tm.applied_score,
        conceptual=tm.conceptual_score,
        doctoral=tm.doctoral_score,
    )
