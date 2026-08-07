"""
cerebrum_core.database.study_plan_registry.weeks
====================================================
plan_week_registry / plan_day_registry / plan_task_registry reads and
writes. This is the new layer: everything a densified phase's
day-by-day detail needs, plus the read paths the Plan Progress widget
and the auto-completion sweep depend on.

Deliberately topic-agnostic about *scoring* — this mixin only knows
about tasks/status/topic strings. Combining topic with topic_mastery
(which lives in note_registry.db, a separate file) is the service
layer's job, not this one's — see study_plan_progress_service.py.
"""

from __future__ import annotations

import json
from typing import Optional


class WeeksMixin:
    # --------------------------------------------------
    # Writes — bulk insert from PHASE_WEEKS_SCHEMA output
    # --------------------------------------------------
    def insert_weeks_inator(self, plan_id: str, phase_id: int, weeks: list[dict]):
        """
        weeks is the "weeks" array from a PHASE_WEEKS_SCHEMA-conformant
        LLM response: [{week_number, focus_summary, topics, days: [...]}]

        Re-runnable: an existing week at the same (plan_id, week_number)
        is deleted first (cascades to its days/tasks via ON DELETE
        CASCADE) before the fresh insert — so re-densifying a phase, or
        regenerating a single week after a replan, doesn't leave stale
        duplicate rows behind. Any manual completion state on the old
        week's tasks is intentionally discarded; callers that need to
        preserve completed-task history across a redensify should read
        it out via fetch_week_inator before calling this.

        Returns the list of new week_ids in the same order as `weeks`.
        """
        new_week_ids = []
        with self._lock:
            conn = self._get_conn()
            try:
                for week in weeks:
                    week_number = week["week_number"]

                    conn.execute(
                        """
                        DELETE FROM plan_week_registry
                        WHERE plan_id = ? AND week_number = ?
                        """,
                        (plan_id, week_number),
                    )

                    cur = conn.execute(
                        """
                        INSERT INTO plan_week_registry
                            (plan_id, phase_id, week_number, focus_summary, topics_json, status)
                        VALUES (?, ?, ?, ?, ?, 'pending')
                        """,
                        (
                            plan_id,
                            phase_id,
                            week_number,
                            week.get("focus_summary"),
                            json.dumps(week.get("topics", [])),
                        ),
                    )
                    week_id = cur.lastrowid
                    new_week_ids.append(week_id)

                    for day in week.get("days", []):
                        cur = conn.execute(
                            """
                            INSERT INTO plan_day_registry (week_id, day_of_week)
                            VALUES (?, ?)
                            """,
                            (week_id, day["day_of_week"]),
                        )
                        day_id = cur.lastrowid

                        for task in day.get("tasks", []):
                            conn.execute(
                                """
                                INSERT INTO plan_task_registry
                                    (day_id, label, task_type, topic, target_minutes, source_hint)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    day_id,
                                    task["label"],
                                    task["task_type"],
                                    task.get("topic"),
                                    task.get("target_minutes"),
                                    task.get("source_hint"),
                                ),
                            )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return new_week_ids

    # --------------------------------------------------
    # Status
    # --------------------------------------------------
    def mark_week_status_inator(self, week_id: int, status: str):
        VALID = {"pending", "active", "complete"}
        if status not in VALID:
            raise ValueError("Invalid week status")
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE plan_week_registry SET status = ? WHERE week_id = ?",
                    (status, week_id),
                )
                conn.commit()
            finally:
                conn.close()

    def complete_task_inator(self, task_id: int, auto_resolved: bool = False):
        """auto_resolved=True marks a task completed by the engram-attempt
        sweep (see study_plan_progress_service.sweep_auto_complete), not
        by a user tap — kept distinct so the UI can show "auto-detected"
        vs "you marked this" differently if useful later."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE plan_task_registry
                    SET status = 'complete',
                        completed_at = CURRENT_TIMESTAMP,
                        auto_resolved = ?
                    WHERE task_id = ?
                    """,
                    (int(auto_resolved), task_id),
                )
                conn.commit()
            finally:
                conn.close()

    def reopen_task_inator(self, task_id: int):
        """Undo — lets a user un-check a task they tapped by mistake."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE plan_task_registry
                    SET status = 'pending', completed_at = NULL, auto_resolved = 0
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )
                conn.commit()
            finally:
                conn.close()

    # --------------------------------------------------
    # Reads
    # --------------------------------------------------
    def _fetch_week_with_days(self, conn, week_row) -> dict:
        week = dict(week_row)
        week["topics"] = json.loads(week.pop("topics_json") or "[]")

        day_rows = conn.execute(
            """
            SELECT day_id, day_of_week FROM plan_day_registry
            WHERE week_id = ? ORDER BY day_of_week ASC
            """,
            (week["week_id"],),
        ).fetchall()

        days = []
        for day_row in day_rows:
            day = dict(day_row)
            task_rows = conn.execute(
                """
                SELECT task_id, label, task_type, topic, target_minutes,
                       source_hint, status, auto_resolved, completed_at
                FROM plan_task_registry
                WHERE day_id = ? ORDER BY task_id ASC
                """,
                (day["day_id"],),
            ).fetchall()
            day["tasks"] = [dict(t) for t in task_rows]
            days.append(day)

        week["days"] = days
        return week

    def fetch_week_inator(self, week_id: int) -> Optional[dict]:
        conn = self._get_conn()
        try:
            week_row = conn.execute(
                "SELECT * FROM plan_week_registry WHERE week_id = ?", (week_id,)
            ).fetchone()
            if not week_row:
                return None
            return self._fetch_week_with_days(conn, week_row)
        finally:
            conn.close()

    def fetch_current_week_inator(self, plan_id: str) -> Optional[dict]:
        """The week with status='active' for this plan. There should be
        at most one — the service layer is responsible for advancing
        which week is active as time/completion progresses; this is a
        pure read."""
        conn = self._get_conn()
        try:
            week_row = conn.execute(
                """
                SELECT * FROM plan_week_registry
                WHERE plan_id = ? AND status = 'active'
                ORDER BY week_number ASC LIMIT 1
                """,
                (plan_id,),
            ).fetchone()
            if not week_row:
                return None
            return self._fetch_week_with_days(conn, week_row)
        finally:
            conn.close()

    def fetch_weeks_for_phase_inator(self, plan_id: str, phase_id: int) -> list[dict]:
        """All weeks generated for a phase so far (a phase may not be
        fully densified — this returns whatever exists, which is the
        whole point of the active-phase-only generation strategy)."""
        conn = self._get_conn()
        try:
            week_rows = conn.execute(
                """
                SELECT * FROM plan_week_registry
                WHERE plan_id = ? AND phase_id = ?
                ORDER BY week_number ASC
                """,
                (plan_id, phase_id),
            ).fetchall()
            return [self._fetch_week_with_days(conn, w) for w in week_rows]
        finally:
            conn.close()

    def fetch_pending_topic_tasks_inator(self, plan_id: str) -> list[dict]:
        """practice/review tasks (the auto-resolvable types) that are
        still pending, across every week of this plan — for the
        engram-attempt sweep to check against and bulk-resolve. Returns
        enough context (day_of_week's parent week) for the sweep to
        scope its date-window query per task."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT t.task_id, t.topic, t.task_type,
                       d.day_of_week, w.week_id, w.week_number, w.plan_id
                FROM plan_task_registry t
                JOIN plan_day_registry d ON d.day_id = t.day_id
                JOIN plan_week_registry w ON w.week_id = d.week_id
                WHERE w.plan_id = ?
                  AND t.status = 'pending'
                  AND t.task_type IN ('practice', 'review')
                  AND t.topic IS NOT NULL
                """,
                (plan_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_week_task_counts_inator(self, week_id: int) -> dict:
        """Cheap progress signal that needs no cross-db data: raw task
        completion ratio for a week. The service layer layers
        topic_mastery on top of this for the richer readiness score."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) AS completed
                FROM plan_task_registry t
                JOIN plan_day_registry d ON d.day_id = t.day_id
                WHERE d.week_id = ?
                """,
                (week_id,),
            ).fetchone()
        finally:
            conn.close()
        total = row["total"] or 0
        completed = row["completed"] or 0
        return {
            "total": total,
            "completed": completed,
            "ratio": (completed / total) if total else 0.0,
        }
