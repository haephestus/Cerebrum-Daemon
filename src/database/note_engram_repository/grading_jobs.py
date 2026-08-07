"""
cerebrum_core.engrams.storage.note_engram_repository.grading_jobs
=====================================================================
Backs grading.worker.SQLiteWorkerLoop: CRUD on grading_jobs plus the
join needed to reassemble a GradingJobPayload for one attempt, so the
worker doesn't have to know the engram_attempts -> engrams -> notes ->
long_question_responses join itself.

Kept separate from generation_queue.py even though both are "a job
queue table" — they serve different consumers (grading worker vs.
generation pipeline) and share no code, so merging them would just mean
hunting through one bigger file to find either.
"""

from __future__ import annotations

from typing import Optional


class GradingJobsMixin:
    def create_grading_job(self, attempt_id: str, priority: int = 5) -> str:
        """Enqueue a grading job for an attempt and return its row id so the
        caller can hand it back to the client for polling."""
        from ._base import _id

        job_id = _id()
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO grading_jobs (id, attempt_id, status, priority) VALUES (?, ?, 'pending', ?)",
                (job_id, attempt_id, priority),
            )
            conn.commit()
        finally:
            conn.close()
        return job_id

    def update_grading_job(
        self, job_id: str, status: str, error: Optional[str] = None
    ) -> None:
        conn = self._get_conn()
        try:
            if status == "processing":
                conn.execute(
                    "UPDATE grading_jobs SET status=?, started_at=datetime('now'), attempts=attempts+1 WHERE id=?",
                    (status, job_id),
                )
            elif status == "done":
                conn.execute(
                    "UPDATE grading_jobs SET status=?, completed_at=datetime('now') WHERE id=?",
                    (status, job_id),
                )
            else:
                conn.execute(
                    "UPDATE grading_jobs SET status=?, error=? WHERE id=?",
                    (status, error, job_id),
                )
            conn.commit()
        finally:
            conn.close()

    def fetch_pending_grading_jobs(self, limit: int = 10) -> list[dict]:
        """Up to `limit` pending grading_jobs rows, highest priority first
        (ties broken oldest-first). Each row is a dict with keys: job_id,
        attempt_id, attempts, priority, created_at.

        `id` is aliased to `job_id` here because worker.run_worker_batch /
        WorkerLoop read job_row["job_id"], matching the same convention
        fetch_pending_generation_jobs uses for the generation queue.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id AS job_id, attempt_id, attempts, priority, created_at
                FROM grading_jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_grading_job(self, job_id: str) -> Optional[dict]:
        """One grading_jobs row joined to its attempt, for status polling.
        Returns job fields plus the attempt's owner/score/grader so a status
        endpoint can both authorise the caller (attempt_user_id) and hand back
        the grade once status is 'done'. None if the job doesn't exist."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT gj.id           AS job_id,
                       gj.attempt_id   AS attempt_id,
                       gj.status       AS status,
                       gj.priority     AS priority,
                       gj.attempts     AS attempts,
                       gj.error        AS error,
                       gj.created_at   AS created_at,
                       gj.started_at   AS started_at,
                       gj.completed_at AS completed_at,
                       ea.user_id      AS attempt_user_id,
                       ea.score        AS attempt_score,
                       ea.grader       AS attempt_grader
                FROM grading_jobs gj
                JOIN engram_attempts ea ON ea.id = gj.attempt_id
                WHERE gj.id = ?
                """,
                (job_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_grading_context(self, attempt_id: str) -> Optional[dict]:
        """Joins engram_attempts -> engrams -> notes ->
        long_question_responses for one attempt. Returns a dict with keys
        engram_id, user_id, target_cognitive_level, note_id, topic, raw_answer, or
        None if the attempt doesn't exist. The parsed LongQuestionContent
        itself is deliberately NOT reassembled here — get_engram() already
        knows how to join long_question_content + long_question_parts, so
        the caller (SQLiteWorkerLoop.hydrate_job) fetches that separately
        rather than duplicating the logic.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT ea.engram_id      AS engram_id,
                       ea.user_id        AS user_id,
                       ea.target_cognitive_level AS target_cognitive_level,
                       e.note_id         AS note_id,
                       n.topic           AS topic,
                       lqr.raw_answer    AS raw_answer
                FROM engram_attempts ea
                JOIN engrams e ON e.id = ea.engram_id
                LEFT JOIN notes n ON n.id = e.note_id
                LEFT JOIN long_question_responses lqr ON lqr.attempt_id = ea.id
                WHERE ea.id = ?
                """,
                (attempt_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_short_question_grading_context(self, attempt_id: str) -> Optional[dict]:
        """Scalar grading context for a short_question attempt: engram_id,
        user_id, target_cognitive_level, note_id, topic. Returns None if the
        attempt doesn't exist.

        Deliberately does NOT join in the raw answers or the question content:
        short_question answers are one-row-per-sub-question (fetched via
        get_short_question_responses) and the parsed QuizContent is
        reassembled by get_engram — same division of labour as
        get_grading_context uses for long questions.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT ea.engram_id      AS engram_id,
                       ea.user_id        AS user_id,
                       ea.target_cognitive_level AS target_cognitive_level,
                       e.note_id         AS note_id,
                       n.topic           AS topic
                FROM engram_attempts ea
                JOIN engrams e ON e.id = ea.engram_id
                LEFT JOIN notes n ON n.id = e.note_id
                WHERE ea.id = ?
                """,
                (attempt_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
