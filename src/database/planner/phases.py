"""
database.study_plan_registry.phases
=====================================================
plan_phase_registry reads/writes. Migrated verbatim from the
pre-split StudyPlanRegisterInator.
"""

from __future__ import annotations

import json


class PhasesMixin:
    def mark_phase_status_inator(self, plan_id: str, phase_id: int, status: str):
        VALID_STATUSES = {"not_started", "in_progress", "completed", "skipped"}
        if status not in VALID_STATUSES:
            raise ValueError("Invalid phase status")

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE plan_phase_registry
                    SET status = ?,
                        completed_at = CASE WHEN ? = 'completed'
                                            THEN CURRENT_TIMESTAMP
                                            ELSE completed_at END
                    WHERE plan_id = ? AND phase_id = ?
                    """,
                    (status, status, plan_id, phase_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def fetch_incomplete_phases_inator(self, plan_id: str):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT phase_id, phase_label, month_start, month_end,
                       theme, milestone, tracks_json AS tracks, status
                FROM plan_phase_registry
                WHERE plan_id = ? AND status != 'completed'
                ORDER BY phase_id ASC
                """,
                (plan_id,),
            ).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            record = dict(row)
            record["tracks"] = json.loads(record["tracks"]) if record["tracks"] else {}
            results.append(record)
        return results

    def fetch_phase_inator(self, plan_id: str, phase_id: int):
        """Single phase lookup — needed by densify_phase to pull the
        phase's tracks/theme as generation context without fetching
        every phase in the plan."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT phase_id, phase_label, month_start, month_end,
                       theme, milestone, tracks_json AS tracks, status
                FROM plan_phase_registry
                WHERE plan_id = ? AND phase_id = ?
                """,
                (plan_id, phase_id),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        record = dict(row)
        record["tracks"] = json.loads(record["tracks"]) if record["tracks"] else {}
        return record
