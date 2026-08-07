"""
cerebrum_core.database.study_plan_registry.metrics
======================================================
plan_success_metric_registry reads/writes. Migrated verbatim from the
pre-split StudyPlanRegisterInator.
"""

from __future__ import annotations


class MetricsMixin:
    def mark_metric_achieved_inator(self, metric_row_id: int):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE plan_success_metric_registry
                    SET achieved = 1, achieved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (metric_row_id,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def fetch_unachieved_metrics_inator(self, plan_id: str):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, month_marker, checkpoint, is_binary_check
                FROM plan_success_metric_registry
                WHERE plan_id = ? AND achieved = 0
                ORDER BY id ASC
                """,
                (plan_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
