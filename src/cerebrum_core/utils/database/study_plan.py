"""
cerebrum_core.utils.database.study_plan_registry
===============================================
allows you to register, track, and update study plans as they are
generated and revised
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from cerebrum_core.utils.file_util_inator import CerebrumPaths


class StudyPlanRegisterInator:
    _lock = threading.Lock()

    def __init__(self, db_path: str = "registry/study_plan_registry.db"):
        self.DB_PATH = CerebrumPaths().kb_root_dir() / db_path
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._table_initiator_inator()

    # --------------------------------------------------
    # Connection helper
    # --------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.DB_PATH, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # --------------------------------------------------
    # Table setup
    # --------------------------------------------------
    def _table_initiator_inator(self):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS study_plan_registry (
                    id INTEGER PRIMARY KEY,
                    plan_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    target_role TEXT,
                    total_duration_months INTEGER,
                    guiding_principle TEXT,
                    status TEXT DEFAULT 'draft',
                    version INTEGER DEFAULT 1,
                    superseded_by_plan_id TEXT,
                    starting_position_json TEXT,
                    weekly_rhythm_json TEXT,
                    immediate_next_actions_json TEXT,
                    raw_plan_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_study_plan_registry_plan_id
                ON study_plan_registry(plan_id)
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_phase_registry (
                    id INTEGER PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    phase_id INTEGER NOT NULL,
                    phase_label TEXT,
                    month_start INTEGER,
                    month_end INTEGER,
                    theme TEXT,
                    milestone TEXT,
                    tracks_json TEXT,
                    status TEXT DEFAULT 'not_started',
                    completed_at TIMESTAMP,
                    FOREIGN KEY (plan_id) REFERENCES study_plan_registry(plan_id)
                        ON DELETE CASCADE,
                    UNIQUE (plan_id, phase_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_success_metric_registry (
                    id INTEGER PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    phase_id INTEGER,
                    month_marker TEXT,
                    checkpoint TEXT,
                    is_binary_check INTEGER DEFAULT 1,
                    achieved INTEGER DEFAULT 0,
                    achieved_at TIMESTAMP,
                    FOREIGN KEY (plan_id) REFERENCES study_plan_registry(plan_id)
                        ON DELETE CASCADE
                )
                """
            )

            conn.commit()
            conn.close()

    # --------------------------------------------------
    # Register plan (full write from LLM output)
    # --------------------------------------------------
    def register_inator(
        self,
        plan_id: str,
        user_id: Optional[str],
        plan_data: dict,
    ):
        """
        plan_data is the raw dict conforming to STUDY_PLAN_SCHEMA
        (plan_overview, phases, weekly_rhythm, success_metrics,
        immediate_next_actions).
        """
        overview = plan_data.get("plan_overview", {})

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO study_plan_registry (
                        plan_id,
                        user_id,
                        target_role,
                        total_duration_months,
                        guiding_principle,
                        starting_position_json,
                        weekly_rhythm_json,
                        immediate_next_actions_json,
                        raw_plan_json
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
                            plan_id,
                            phase_id,
                            phase_label,
                            month_start,
                            month_end,
                            theme,
                            milestone,
                            tracks_json
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
                            plan_id,
                            month_marker,
                            checkpoint,
                            is_binary_check
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
        """
        Parses strings like '0-3' or 'Month 0 to Month 3' into (start, end)
        integers. Falls back to (None, None) if unparseable.
        """
        import re

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
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    UPDATE study_plan_registry
                    SET status = ?,
                        last_updated = CURRENT_TIMESTAMP
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

    def mark_phase_status_inator(self, plan_id: str, phase_id: int, status: str):
        VALID_STATUSES = {"not_started", "in_progress", "completed", "skipped"}
        if status not in VALID_STATUSES:
            raise ValueError("Invalid phase status")

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
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

    def mark_metric_achieved_inator(self, metric_row_id: int):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    UPDATE plan_success_metric_registry
                    SET achieved = 1,
                        achieved_at = CURRENT_TIMESTAMP
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

    def bump_version_inator(self, old_plan_id: str, new_plan_id: str):
        """Marks old_plan_id as archived and links it forward to new_plan_id."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    UPDATE study_plan_registry
                    SET status = 'archived',
                        superseded_by_plan_id = ?,
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
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT raw_plan_json
                FROM study_plan_registry
                WHERE plan_id = ?
                """,
                (plan_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        return json.loads(row[0]) if row and row[0] else None

    def fetch_stale_plans_inator(self, grace_days: int = 0):
        """
        Returns active plans whose total_duration_months has elapsed
        (based on created_at) but which are not yet completed/archived.
        Useful for a background job flagging plans due for review/revision.

        grace_days: extra buffer added past the duration before a plan
                    is considered stale (e.g. avoid flagging on day one
                    of overrun).
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                    SELECT
                        plan_id,
                        user_id,
                        target_role,
                        total_duration_months,
                        status,
                        version,
                        created_at,
                        CAST(julianday('now') - julianday(created_at) AS INTEGER) AS days_elapsed
                    FROM study_plan_registry
                    WHERE status = 'active'
                      AND (julianday('now') - julianday(created_at))
                          > (total_duration_months * 30.44 + ?)
                    """,
                (grace_days,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        columns = [
            "plan_id",
            "user_id",
            "target_role",
            "total_duration_months",
            "status",
            "version",
            "created_at",
            "days_elapsed",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def fetch_active_plans_inator(self, user_id: Optional[str] = None):
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            if user_id:
                cursor.execute(
                    """
                    SELECT plan_id, user_id, target_role, total_duration_months, version
                    FROM study_plan_registry
                    WHERE status = 'active' AND user_id = ?
                    """,
                    (user_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT plan_id, user_id, target_role, total_duration_months, version
                    FROM study_plan_registry
                    WHERE status = 'active'
                    """
                )
            rows = cursor.fetchall()
        finally:
            conn.close()

        columns = [
            "plan_id",
            "user_id",
            "target_role",
            "total_duration_months",
            "version",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def fetch_incomplete_phases_inator(self, plan_id: str):
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT phase_id, phase_label, month_start, month_end,
                       theme, milestone, tracks_json, status
                FROM plan_phase_registry
                WHERE plan_id = ? AND status != 'completed'
                ORDER BY phase_id ASC
                """,
                (plan_id,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        columns = [
            "phase_id",
            "phase_label",
            "month_start",
            "month_end",
            "theme",
            "milestone",
            "tracks",
            "status",
        ]
        results = []
        for row in rows:
            record = dict(zip(columns, row))
            record["tracks"] = json.loads(record["tracks"]) if record["tracks"] else {}
            results.append(record)
        return results

    def fetch_unachieved_metrics_inator(self, plan_id: str):
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT id, month_marker, checkpoint, is_binary_check
                FROM plan_success_metric_registry
                WHERE plan_id = ? AND achieved = 0
                ORDER BY id ASC
                """,
                (plan_id,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        columns = ["id", "month_marker", "checkpoint", "is_binary_check"]
        return [dict(zip(columns, row)) for row in rows]

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def check_inator(self, plan_id: str, field: str = "") -> bool:
        VALID_FIELDS = {"status", "version"}

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            if field:
                if field not in VALID_FIELDS:
                    raise ValueError("Invalid field requested")
                cursor.execute(
                    f"""
                    SELECT {field}
                    FROM study_plan_registry
                    WHERE plan_id = ?
                    """,
                    (plan_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT 1
                    FROM study_plan_registry
                    WHERE plan_id = ?
                    """,
                    (plan_id,),
                )

            result = cursor.fetchone()
        finally:
            conn.close()

        return bool(result and (result[0] if field else True))

    def show_all_inator(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT id, plan_id, user_id, target_role, total_duration_months,
                       status, version, superseded_by_plan_id, created_at, last_updated
                FROM study_plan_registry
                """
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        columns = [
            "id",
            "plan_id",
            "user_id",
            "target_role",
            "total_duration_months",
            "status",
            "version",
            "superseded_by_plan_id",
            "created_at",
            "last_updated",
        ]
        return [dict(zip(columns, row)) for row in rows]

    # --------------------------------------------------
    # Delete / Reset
    # --------------------------------------------------
    def remove_inator(self, plan_id: str):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "DELETE FROM plan_success_metric_registry WHERE plan_id = ?",
                    (plan_id,),
                )
                cursor.execute(
                    "DELETE FROM plan_phase_registry WHERE plan_id = ?",
                    (plan_id,),
                )
                cursor.execute(
                    "DELETE FROM study_plan_registry WHERE plan_id = ?",
                    (plan_id,),
                )

                if cursor.rowcount == 0:
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
            cursor = conn.cursor()

            try:
                if plan_id:
                    cursor.execute(
                        """
                        UPDATE plan_phase_registry
                        SET status = 'not_started', completed_at = NULL
                        WHERE plan_id = ?
                        """,
                        (plan_id,),
                    )
                    cursor.execute(
                        """
                        UPDATE plan_success_metric_registry
                        SET achieved = 0, achieved_at = NULL
                        WHERE plan_id = ?
                        """,
                        (plan_id,),
                    )
                else:
                    cursor.execute(
                        "UPDATE plan_phase_registry SET status = 'not_started', completed_at = NULL"
                    )
                    cursor.execute(
                        "UPDATE plan_success_metric_registry SET achieved = 0, achieved_at = NULL"
                    )

                conn.commit()
                count = cursor.rowcount
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return count
