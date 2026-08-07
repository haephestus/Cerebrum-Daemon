"""
cerebrum_core.engrams.storage.note_engram_repository.generation_queue
=========================================================================
CRUD on engram_generation_queue: the pipeline that decides "generate an
mcq for note X at level Y" queues a row here, and a worker (separate
from the grading worker in grading_jobs.py) drains it via
fetch_pending_generation_jobs.
"""

from __future__ import annotations

from typing import Optional

from ._base import _id


class GenerationQueueMixin:
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
    ) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engram_generation_queue
                  (id, bubble_id, note_id, user_id, trigger, trigger_ref, target_cognitive_level, target_type, instructions)
                VALUES (?,?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _id(),
                    bubble_id,
                    note_id,
                    user_id,
                    trigger,
                    trigger_ref,
                    target_cognitive_level,
                    target_type,
                    instructions,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_pending_generation_jobs(self, limit: int = 10) -> list[dict]:
        """
        Returns up to `limit` pending rows from engram_generation_queue,
        oldest first. Each row is a plain dict with keys:
            id, note_id, user_id, trigger, trigger_ref,
            target_cognitive_level, target_type, instructions, status, created_at
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM engram_generation_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def mark_generation_job_done(self, job_id: str) -> None:
        """Mark a generation queue row as successfully completed."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE engram_generation_queue SET status = 'done' WHERE id = ?",
                    (job_id,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def mark_generation_job_failed(
        self, job_id: str, error: str, retry: bool = True, max_attempts: int = 3
    ) -> None:
        """Mark a job as failed. If retry and attempts < max_attempts, bump
        attempts and reset status to 'pending' so the next poll picks it up
        again; otherwise mark permanently 'failed'."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT attempts FROM engram_generation_queue WHERE id = ?",
                    (job_id,),
                ).fetchone()
                attempts = (
                    int(row["attempts"]) if row and row["attempts"] is not None else 0
                ) + 1
                new_status = (
                    "pending" if (retry and attempts < max_attempts) else "failed"
                )
                conn.execute(
                    """
                    UPDATE engram_generation_queue
                    SET status = ?, attempts = ?,
                        instructions = COALESCE(instructions, '') || ?
                    WHERE id = ?
                    """,
                    (
                        new_status,
                        attempts,
                        f"\n[ERROR attempt {attempts}] {error}",
                        job_id,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
