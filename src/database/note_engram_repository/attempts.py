"""
cerebrum_core.engrams.storage.note_engram_repository.attempts
=================================================================
Everything on engram_attempts plus the four per-type response tables
(mcq_responses, flashcard_responses, short_question_responses,
long_question_responses). These stay together because a "response" only
ever exists in the context of an attempt — save_*_response always takes
an attempt_id, and get_recent_attempt_scores/get_long_question_responses
both read back through engram_attempts.
"""

from __future__ import annotations

import json

from cerebrum_core.engrams.core.types import (
    DimensionScores,
    EngramAttempt,
    FlashcardResponse,
    GraderType,
    LongQuestionResponse,
    MCQResponse,
    QuizResponse,
)


class AttemptsMixin:
    # -----------------------------------------------------------------------
    # Attempts
    # -----------------------------------------------------------------------

    def create_attempt(self, attempt: EngramAttempt) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engram_attempts
                  (id, engram_id, user_id, attempted_at, score, grader,
                   time_spent_ms, note_version, target_cognitive_level, context_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.id,
                    attempt.engram_id,
                    attempt.user_id,
                    attempt.attempted_at,
                    attempt.score,
                    attempt.grader.value,
                    attempt.time_spent_ms,
                    attempt.note_version,
                    attempt.target_cognitive_level,
                    (
                        json.dumps(attempt.context_snapshot)
                        if attempt.context_snapshot
                        else None
                    ),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Atomic submissions
    #
    # Recording an async-graded attempt is three inserts across three tables
    # (engram_attempts, the per-type response table, grading_jobs). Done as
    # separate per-call commits, a crash between them orphans rows — an
    # attempt with no job (never graded) or a job pointing at nothing. These
    # composite methods run all three in one _transaction so the submission
    # either fully lands or not at all. They mirror the SQL of create_attempt
    # / save_*_response / create_grading_job intentionally, since those each
    # open their own connection and can't be composed into one transaction.
    # -----------------------------------------------------------------------

    def _insert_attempt(self, conn, attempt: EngramAttempt) -> None:
        conn.execute(
            """
            INSERT INTO engram_attempts
              (id, engram_id, user_id, attempted_at, score, grader,
               time_spent_ms, note_version, target_cognitive_level, context_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt.id,
                attempt.engram_id,
                attempt.user_id,
                attempt.attempted_at,
                attempt.score,
                attempt.grader.value,
                attempt.time_spent_ms,
                attempt.note_version,
                attempt.target_cognitive_level,
                (
                    json.dumps(attempt.context_snapshot)
                    if attempt.context_snapshot
                    else None
                ),
            ),
        )

    def record_short_question_submission(
        self,
        attempt: EngramAttempt,
        responses: "list[QuizResponse]",
        priority: int = 5,
    ) -> str:
        """Atomically insert the pending attempt, its raw sub-answers, and one
        grading job. Returns the grading job id."""
        from ._base import _id

        job_id = _id()
        with self._transaction() as conn:
            self._insert_attempt(conn, attempt)
            conn.executemany(
                """
                INSERT OR REPLACE INTO short_question_responses
                  (id, attempt_id, question_index, raw_answer,
                   score, is_correct, feedback, misconceptions, graded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.attempt_id,
                        r.question_index,
                        r.raw_answer,
                        r.score,
                        None if r.is_correct is None else int(r.is_correct),
                        r.feedback,
                        json.dumps(r.misconceptions) if r.misconceptions else None,
                        r.graded_at,
                    )
                    for r in responses
                ],
            )
            conn.execute(
                "INSERT INTO grading_jobs (id, attempt_id, status, priority) "
                "VALUES (?, ?, 'pending', ?)",
                (job_id, attempt.id, priority),
            )
        return job_id

    def record_long_question_submission(
        self,
        attempt: EngramAttempt,
        response: LongQuestionResponse,
        priority: int = 5,
    ) -> str:
        """Atomically insert the pending attempt, its raw answer, and one
        grading job. Returns the grading job id."""
        from ._base import _id

        job_id = _id()
        with self._transaction() as conn:
            self._insert_attempt(conn, attempt)
            conn.execute(
                """
                INSERT OR REPLACE INTO long_question_responses
                  (attempt_id, raw_answer, word_count)
                VALUES (?, ?, ?)
                """,
                (response.attempt_id, response.raw_answer, response.word_count),
            )
            conn.execute(
                "INSERT INTO grading_jobs (id, attempt_id, status, priority) "
                "VALUES (?, ?, 'pending', ?)",
                (job_id, attempt.id, priority),
            )
        return job_id

    def update_attempt_score(
        self, attempt_id: str, score: float, grader: GraderType
    ) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE engram_attempts SET score = ?, grader = ? WHERE id = ?",
                (score, grader.value, attempt_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_attempt_scores(
        self, engram_id: str, user_id: str, limit: int = 10
    ) -> list[float]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT score FROM engram_attempts
                WHERE engram_id = ? AND user_id = ? AND score IS NOT NULL
                ORDER BY attempted_at DESC LIMIT ?
                """,
                (engram_id, user_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return list(reversed([float(r["score"]) for r in rows]))

    # -----------------------------------------------------------------------
    # Type-specific responses
    # -----------------------------------------------------------------------

    def save_mcq_response(self, r: MCQResponse) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO mcq_responses
                  (attempt_id, selected_option, correct_option, is_correct, distractor_key)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    r.attempt_id,
                    r.selected_option,
                    r.correct_option,
                    int(r.is_correct),
                    r.distractor_key,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_flashcard_response(self, r: FlashcardResponse) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO flashcard_responses (attempt_id, self_rating, time_to_flip_ms)
                VALUES (?, ?, ?)
                """,
                (r.attempt_id, r.self_rating.value, r.time_to_flip_ms),
            )
            conn.commit()
        finally:
            conn.close()

    def save_short_question_responses(self, responses: list[QuizResponse]) -> None:
        """Persist the student's raw sub-answers for an attempt. Called at
        submit time with score/is_correct/feedback still None — the grading
        worker fills those in later via save_short_question_grades. INSERT OR
        REPLACE keyed on (attempt_id, question_index) so a re-submit of the
        same question within an attempt overwrites rather than duplicates."""
        conn = self._get_conn()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO short_question_responses
                  (id, attempt_id, question_index, raw_answer,
                   score, is_correct, feedback, misconceptions, graded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.attempt_id,
                        r.question_index,
                        r.raw_answer,
                        r.score,
                        None if r.is_correct is None else int(r.is_correct),
                        r.feedback,
                        json.dumps(r.misconceptions) if r.misconceptions else None,
                        r.graded_at,
                    )
                    for r in responses
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def get_short_question_responses(self, attempt_id: str) -> list[QuizResponse]:
        """All sub-question responses for one attempt, ordered by
        question_index. Used by the grading worker to hydrate the raw
        answers it needs to grade, and available for answer-history reads."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM short_question_responses
                WHERE attempt_id = ?
                ORDER BY question_index
                """,
                (attempt_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            QuizResponse(
                id=r["id"],
                attempt_id=r["attempt_id"],
                question_index=r["question_index"],
                raw_answer=r["raw_answer"],
                score=r["score"],
                is_correct=None if r["is_correct"] is None else bool(r["is_correct"]),
                feedback=r["feedback"],
                misconceptions=(
                    json.loads(r["misconceptions"]) if r["misconceptions"] else []
                ),
                graded_at=r["graded_at"],
            )
            for r in rows
        ]

    def save_short_question_grades(self, grades: list[QuizResponse]) -> None:
        """Write AI grades back onto existing response rows for an attempt.
        Matches on (attempt_id, question_index) rather than the row id, since
        the grader works from what it read out of the DB and never mints new
        response rows — it only annotates the ones submit already wrote."""
        conn = self._get_conn()
        try:
            conn.executemany(
                """
                UPDATE short_question_responses
                SET score = ?, is_correct = ?, feedback = ?,
                    misconceptions = ?, graded_at = ?
                WHERE attempt_id = ? AND question_index = ?
                """,
                [
                    (
                        g.score,
                        None if g.is_correct is None else int(g.is_correct),
                        g.feedback,
                        json.dumps(g.misconceptions) if g.misconceptions else None,
                        g.graded_at,
                        g.attempt_id,
                        g.question_index,
                    )
                    for g in grades
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def save_long_question_response(self, r: LongQuestionResponse) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO long_question_responses
                  (attempt_id, raw_answer, word_count, ai_feedback,
                   concepts_demonstrated, concepts_missed, misconceptions,
                   dimension_scores, level_demonstrated, regression_detected,
                   vector_id, graded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.attempt_id,
                    r.raw_answer,
                    r.word_count,
                    r.ai_feedback,
                    (
                        json.dumps(r.concepts_demonstrated)
                        if r.concepts_demonstrated
                        else None
                    ),
                    json.dumps(r.concepts_missed) if r.concepts_missed else None,
                    json.dumps(r.misconceptions) if r.misconceptions else None,
                    (
                        json.dumps(r.dimension_scores.to_dict())
                        if r.dimension_scores
                        else None
                    ),
                    r.level_demonstrated,
                    int(r.regression_detected),
                    r.vector_id,
                    r.graded_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_long_question_responses(
        self, engram_id: str, user_id: str, limit: int = 5
    ) -> list[LongQuestionResponse]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT lqr.* FROM long_question_responses lqr
                JOIN engram_attempts ea ON ea.id = lqr.attempt_id
                WHERE ea.engram_id = ? AND ea.user_id = ? AND lqr.graded_at IS NOT NULL
                ORDER BY ea.attempted_at DESC LIMIT ?
                """,
                (engram_id, user_id, limit),
            ).fetchall()
        finally:
            conn.close()

        result = []
        for row in rows:
            ds_raw = row["dimension_scores"]
            result.append(
                LongQuestionResponse(
                    attempt_id=row["attempt_id"],
                    raw_answer=row["raw_answer"],
                    word_count=row["word_count"] or 0,
                    ai_feedback=row["ai_feedback"],
                    concepts_demonstrated=(
                        json.loads(row["concepts_demonstrated"])
                        if row["concepts_demonstrated"]
                        else []
                    ),
                    concepts_missed=(
                        json.loads(row["concepts_missed"])
                        if row["concepts_missed"]
                        else []
                    ),
                    misconceptions=(
                        json.loads(row["misconceptions"])
                        if row["misconceptions"]
                        else []
                    ),
                    dimension_scores=(
                        DimensionScores.from_dict(json.loads(ds_raw))
                        if ds_raw
                        else None
                    ),
                    level_demonstrated=row["level_demonstrated"],
                    regression_detected=bool(row["regression_detected"]),
                    vector_id=row["vector_id"],
                    graded_at=row["graded_at"],
                )
            )
        return result

    """
    Needed by study_plan_progress_service.sweep_auto_complete: a task's
    completion needs to check "did this user attempt anything under this
    topic within this task's day window", which get_recent_attempt_scores
    can't answer since it's scoped to a single engram_id, not a topic.
    """

    def get_topic_activity_since(
        self, user_id: str, topic: str, since_iso: str, until_iso: str | None = None
    ) -> int:
        """
        Count of engram_attempts by this user, on engrams whose note's
        topic matches, with attempted_at inside [since_iso, until_iso).
        until_iso defaults to now. Same engram->note->topic join as
        _topic_for_engram in mastery.py — kept consistent with that,
        not reinvented here.

        Returns a count rather than a bool so callers can distinguish
        "no activity" from "some activity" if a richer threshold is
        wanted later (e.g. require 2+ attempts, not just 1).
        """
        conn = self._get_conn()
        try:
            # Match on the topic ENTITY (nt.topic_id), resolved from the name's
            # canonical slug, so the planner's topic string still lines up with
            # the note's topic even if the casing/spacing differs.
            topic_id = self._lookup_topic_id(conn, user_id, topic)
            if topic_id is None:
                return 0
            if until_iso:
                rows = conn.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM engram_attempts a
                    JOIN engrams e ON e.id = a.engram_id
                    JOIN notes nt ON nt.id = e.note_id
                    WHERE a.user_id = ?
                      AND nt.topic_id = ?
                      AND a.attempted_at >= ?
                      AND a.attempted_at < ?
                    """,
                    (user_id, topic_id, since_iso, until_iso),
                ).fetchone()
            else:
                rows = conn.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM engram_attempts a
                    JOIN engrams e ON e.id = a.engram_id
                    JOIN notes nt ON nt.id = e.note_id
                    WHERE a.user_id = ?
                      AND nt.topic_id = ?
                      AND a.attempted_at >= ?
                    """,
                    (user_id, topic_id, since_iso),
                ).fetchone()
        finally:
            conn.close()
        return int(rows["n"]) if rows else 0
