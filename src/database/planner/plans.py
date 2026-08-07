"""
database.study_plan_registry.plans
====================================================
Everything that reads/writes study_plan_registry itself: creation from
LLM output, status/version transitions, and plan-level fetchers.

Method bodies are migrated verbatim from the pre-split
StudyPlanRegisterInator — no behavior changes here, just relocated
under the mixin architecture so weeks.py has somewhere consistent to
live alongside it.
"""

from __future__ import annotations

import json
import re
from typing import Optional


class PlansMixin:
    # --------------------------------------------------
    # Register plan (full write from LLM output)
    # --------------------------------------------------
    def register_inator(self, plan_id: str, user_id: Optional[str], plan_data: dict):
        """
        plan_data is the raw dict conforming to STUDY_PLAN_SCHEMA
        (plan_overview, phases, weekly_rhythm, success_metrics,
        immediate_next_actions). Does NOT touch weeks/days/tasks —
        those are written separately by WeeksMixin.insert_weeks_inator
        once a phase is densified.
        """
        overview = plan_data.get("plan_overview", {})

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO study_plan_registry (
                        plan_id, user_id, target_role, total_duration_months,
                        guiding_principle, starting_position_json,
                        weekly_rhythm_json, immediate_next_actions_json, raw_plan_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(plan_id) DO UPDATE SET
                        target_role = excluded.target_role,
                        total_duration_months = excluded.total_duration_months,
                        guiding_principle = excluded.guiding_principle,
                        starting_position_json = excluded.starting_position_json,
                        weekly_rhythm_json = excluded.weekly_rhythm_json,
                        immediate_next_actions_json = excluded.immediate_next_actions_json,
                        raw_plan_json = excluded.raw_plan_json,
                        last_updated = CURRENT_TIMESTAMP
                    """,
                    (
                        plan_id,
                        user_id,
                        overview.get("target_role"),
                        overview.get("total_duration_months"),
                        overview.get("guiding_principle"),
                        json.dumps(overview.get("starting_position", [])),
                        json.dumps(plan_data.get("weekly_rhythm", [])),
                        json.dumps(plan_data.get("immediate_next_actions", [])),
                        json.dumps(plan_data),
                    ),
                )

                for phase in plan_data.get("phases", []):
                    month_start, month_end = self._parse_month_range_inator(
                        phase.get("month_range", "")
                    )
                    cursor.execute(
                        """
                        INSERT INTO plan_phase_registry (
                            plan_id, phase_id, phase_label, month_start, month_end,
                            theme, milestone, tracks_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(plan_id, phase_id) DO UPDATE SET
                            phase_label = excluded.phase_label,
                            month_start = excluded.month_start,
                            month_end = excluded.month_end,
                            theme = excluded.theme,
                            milestone = excluded.milestone,
                            tracks_json = excluded.tracks_json
                        """,
                        (
                            plan_id,
                            phase.get("phase_id"),
                            phase.get("phase_label"),
                            month_start,
                            month_end,
                            phase.get("theme"),
                            phase.get("milestone"),
                            json.dumps(phase.get("tracks", {})),
                        ),
                    )

                for metric in plan_data.get("success_metrics", []):
                    cursor.execute(
                        """
                        INSERT INTO plan_success_metric_registry (
                            plan_id, month_marker, checkpoint, is_binary_check
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            plan_id,
                            metric.get("month_marker"),
                            metric.get("checkpoint"),
                            int(metric.get("is_binary_check", True)),
                        ),
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @staticmethod
    def _parse_month_range_inator(month_range: str):
        """Parses '0-3' or 'Month 0 to Month 3' into (start, end) ints."""
        nums = re.findall(r"\d+", month_range or "")
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if len(nums) == 1:
            return int(nums[0]), int(nums[0])
        return None, None

    # --------------------------------------------------
    # Status updates
    # --------------------------------------------------
    def mark_status_inator(self, plan_id: str, status: str):
        VALID_STATUSES = {"draft", "active", "completed", "archived"}
        if status not in VALID_STATUSES:
            raise ValueError("Invalid plan status")

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE study_plan_registry
                    SET status = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE plan_id = ?
                    """,
                    (status, plan_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def bump_version_inator(self, old_plan_id: str, new_plan_id: str):
        """Marks old_plan_id as archived and links it forward to new_plan_id."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE study_plan_registry
                    SET status = 'archived', superseded_by_plan_id = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE plan_id = ?
                    """,
                    (new_plan_id, old_plan_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # --------------------------------------------------
    # Fetchers
    # --------------------------------------------------
    def fetch_plan_inator(self, plan_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT raw_plan_json FROM study_plan_registry WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row and row[0] else None

    def fetch_all_plans_inator(self, user_id: Optional[str] = None):
        """Every plan regardless of status. status is included per-row so
        callers (e.g. the Flutter plans-tab toggle) can filter client-side."""
        conn = self._get_conn()
        try:
            base = """
                SELECT plan_id, user_id, target_role, total_duration_months,
                       status, version, superseded_by_plan_id,
                       created_at, last_updated
                FROM study_plan_registry
            """
            if user_id:
                rows = conn.execute(
                    base + " WHERE user_id = ? ORDER BY last_updated DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(base + " ORDER BY last_updated DESC").fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def fetch_active_plans_inator(self, user_id: Optional[str] = None):
        """Kept for existing callers that specifically want active-only.
        The plans-tab UI should use fetch_all_plans_inator + client-side
        filtering instead — see routes_study_plan.get_user_plans."""
        conn = self._get_conn()
        try:
            base = """
                SELECT plan_id, user_id, target_role, total_duration_months, version
                FROM study_plan_registry WHERE status = 'active'
            """
            if user_id:
                rows = conn.execute(base + " AND user_id = ?", (user_id,)).fetchall()
            else:
                rows = conn.execute(base).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def fetch_stale_plans_inator(self, grace_days: int = 0):
        """Active plans whose total_duration_months has elapsed but which
        aren't completed/archived yet. grace_days buffers past the
        duration before flagging (avoid flagging on day one of overrun)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT plan_id, user_id, target_role, total_duration_months,
                       status, version, created_at,
                       CAST(julianday('now') - julianday(created_at) AS INTEGER) AS days_elapsed
                FROM study_plan_registry
                WHERE status = 'active'
                  AND (julianday('now') - julianday(created_at))
                      > (total_duration_months * 30.44 + ?)
                """,
                (grace_days,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def check_inator(self, plan_id: str, field: str = "") -> bool:
        VALID_FIELDS = {"status", "version"}
        conn = self._get_conn()
        try:
            if field:
                if field not in VALID_FIELDS:
                    raise ValueError("Invalid field requested")
                result = conn.execute(
                    f"SELECT {field} FROM study_plan_registry WHERE plan_id = ?",
                    (plan_id,),
                ).fetchone()
            else:
                result = conn.execute(
                    "SELECT 1 FROM study_plan_registry WHERE plan_id = ?", (plan_id,)
                ).fetchone()
        finally:
            conn.close()
        return bool(result and (result[0] if field else True))

    def show_all_inator(self):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, plan_id, user_id, target_role, total_duration_months,
                       status, version, superseded_by_plan_id, created_at, last_updated
                FROM study_plan_registry
                """
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # --------------------------------------------------
    # Delete / Reset
    # --------------------------------------------------
    def remove_inator(self, plan_id: str):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM plan_success_metric_registry WHERE plan_id = ?",
                    (plan_id,),
                )
                conn.execute(
                    "DELETE FROM plan_phase_registry WHERE plan_id = ?", (plan_id,)
                )
                cur = conn.execute(
                    "DELETE FROM study_plan_registry WHERE plan_id = ?", (plan_id,)
                )
                if cur.rowcount == 0:
                    raise FileNotFoundError("Study plan registry entry not found")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def reset_inator(self, plan_id: Optional[str] = None):
        """Resets all phase/metric progress back to not_started/unachieved."""
        with self._lock:
            conn = self._get_conn()
            try:
                if plan_id:
                    conn.execute(
                        """
                        UPDATE plan_phase_registry
                        SET status = 'not_started', completed_at = NULL
                        WHERE plan_id = ?
                        """,
                        (plan_id,),
                    )
                    cur = conn.execute(
                        """
                        UPDATE plan_success_metric_registry
                        SET achieved = 0, achieved_at = NULL
                        WHERE plan_id = ?
                        """,
                        (plan_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE plan_phase_registry SET status = 'not_started', completed_at = NULL"
                    )
                    cur = conn.execute(
                        "UPDATE plan_success_metric_registry SET achieved = 0, achieved_at = NULL"
                    )
                conn.commit()
                count = cur.rowcount
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return count
