"""
database.note_engram_repository.learning_profile
================================================
Two-layer learning-profile storage:

  learning_profile_declared   -- the user's self-declared axis values (prior).
                                  One row/user; upserted wholesale.
  learning_profile_evidence   -- append-only behavioural signals (evidence).
                                  The inferred posterior is DERIVED from these
                                  by cerebrum_core.learning_profile_inator; there
                                  is no cached posterior here on purpose -- the
                                  log is the single source of truth and the
                                  growth-trajectory record.

Axis names/ranges are owned by cerebrum_core.learning_profile_inator.AXES;
this layer stores whatever axis strings it's handed (validation lives at the
domain edge), so the DB doesn't change when the axis set evolves.
"""

from __future__ import annotations

import json
from typing import Optional

from ._base import _id


class LearningProfileMixin:
    # ------------------------------------------------------------------ #
    # Declared layer (the prior)
    # ------------------------------------------------------------------ #
    def get_declared_profile(self, user_id: str) -> dict:
        """The user's self-declared axis map ({} if they've never set one)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT axes FROM learning_profile_declared WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row["axes"]) if row else {}

    def set_declared_profile(self, user_id: str, axes: dict) -> None:
        """Upsert the declared axis map wholesale. Never touches evidence."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO learning_profile_declared (user_id, axes, updated_at)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT(user_id) DO UPDATE SET
                        axes = excluded.axes,
                        updated_at = excluded.updated_at
                    """,
                    (user_id, json.dumps(axes)),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Inferred layer (the evidence log)
    # ------------------------------------------------------------------ #
    def add_profile_evidence(
        self,
        user_id: str,
        source: str,
        axis: str,
        value: float,
        weight: float = 1.0,
        ref: Optional[str] = None,
    ) -> None:
        """Append a single behavioural signal to the evidence log."""
        self.add_profile_evidence_batch(
            user_id,
            [{"source": source, "axis": axis, "value": value, "weight": weight, "ref": ref}],
        )

    def add_profile_evidence_batch(self, user_id: str, rows: list[dict]) -> None:
        """Append many signals at once. Each row: source, axis, value,
        optional weight (default 1.0), optional ref."""
        if not rows:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executemany(
                    """
                    INSERT INTO learning_profile_evidence
                        (id, user_id, source, axis, value, weight, ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            _id(),
                            user_id,
                            r["source"],
                            r["axis"],
                            float(r["value"]),
                            float(r.get("weight", 1.0)),
                            r.get("ref"),
                        )
                        for r in rows
                    ],
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def replace_evidence_for_ref(
        self, user_id: str, source: str, ref: str, rows: list[dict]
    ) -> None:
        """Atomically replace all evidence for (user_id, source, ref) with
        `rows`. Makes inference idempotent: re-analysing the same note (or
        recomputing perf) overwrites its signal instead of piling up duplicates.
        `rows` items: source, axis, value, optional weight, optional ref."""
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM learning_profile_evidence "
                "WHERE user_id = ? AND source = ? AND ref = ?",
                (user_id, source, ref),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO learning_profile_evidence
                        (id, user_id, source, axis, value, weight, ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            _id(),
                            user_id,
                            r["source"],
                            r["axis"],
                            float(r["value"]),
                            float(r.get("weight", 1.0)),
                            r.get("ref"),
                        )
                        for r in rows
                    ],
                )

    def get_profile_evidence(
        self, user_id: str, axis: Optional[str] = None
    ) -> list[dict]:
        """The evidence log for a user, oldest first (so callers can also read
        it as a growth trajectory). Optionally filtered to one axis."""
        conn = self._get_conn()
        try:
            if axis:
                rows = conn.execute(
                    "SELECT * FROM learning_profile_evidence "
                    "WHERE user_id = ? AND axis = ? ORDER BY created_at",
                    (user_id, axis),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM learning_profile_evidence "
                    "WHERE user_id = ? ORDER BY created_at",
                    (user_id,),
                ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
