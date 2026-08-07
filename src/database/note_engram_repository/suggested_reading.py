"""
database.note_engram_repository.suggested_reading
=================================================
Persistence for suggested readings (gap 3). Candidates are written per seed
(a note_id / topic); the domain orchestrator (cerebrum_core.suggested_reading_inator)
produces them and this mixin stores/lists/updates them.

`replace_candidate_suggestions` only replaces rows still in the 'candidate'
state, so a re-run refreshes candidates without wiping anything the user has
already accepted/dismissed or that has been ingested into the KB.
"""

from __future__ import annotations

import json
from typing import Optional

from ._base import _id


class SuggestedReadingMixin:
    def replace_candidate_suggestions(
        self, user_id: str, seed_ref: str, readings: list[dict]
    ) -> None:
        """Replace this seed's *candidate* suggestions with `readings`
        (SuggestedReading-shaped dicts). Accepted/ingested/dismissed rows for
        the same seed are left untouched."""
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM suggested_readings "
                "WHERE user_id = ? AND seed_ref = ? AND status = 'candidate'",
                (user_id, seed_ref),
            )
            if readings:
                conn.executemany(
                    """
                    INSERT INTO suggested_readings
                        (id, user_id, seed_ref, title, source, url,
                         file_fingerprint, license, snippet, reason,
                         addresses, score, in_kb, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate')
                    """,
                    [
                        (
                            _id(),
                            user_id,
                            seed_ref,
                            r["title"],
                            r["source"],
                            r.get("url"),
                            r.get("file_fingerprint"),
                            r.get("license"),
                            r.get("snippet"),
                            r.get("reason"),
                            json.dumps(r.get("addresses") or []),
                            float(r.get("score", 0.0)),
                            1 if r.get("in_kb") else 0,
                        )
                        for r in readings
                    ],
                )

    def list_suggestions(
        self, user_id: str, seed_ref: Optional[str] = None
    ) -> list[dict]:
        """List a user's suggestions (optionally for one seed), best score
        first. `addresses` is parsed back to a list."""
        conn = self._get_conn()
        try:
            if seed_ref:
                rows = conn.execute(
                    "SELECT * FROM suggested_readings "
                    "WHERE user_id = ? AND seed_ref = ? ORDER BY score DESC",
                    (user_id, seed_ref),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM suggested_readings "
                    "WHERE user_id = ? ORDER BY created_at DESC, score DESC",
                    (user_id,),
                ).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["addresses"] = json.loads(d.get("addresses") or "[]")
            d["in_kb"] = bool(d.get("in_kb"))
            out.append(d)
        return out

    def get_suggestion(self, suggestion_id: str) -> Optional[dict]:
        """One suggestion by id (carries user_id for ownership checks)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM suggested_readings WHERE id = ?", (suggestion_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        d = dict(row)
        d["addresses"] = json.loads(d.get("addresses") or "[]")
        d["in_kb"] = bool(d.get("in_kb"))
        return d

    def set_suggestion_status(
        self,
        suggestion_id: str,
        status: str,
        in_kb: Optional[bool] = None,
        file_fingerprint: Optional[str] = None,
    ) -> None:
        """Update a suggestion's lifecycle state (used by Phase-1 accept/ingest
        and dismiss). Only sets in_kb / file_fingerprint when provided."""
        sets = ["status = ?"]
        params: list = [status]
        if in_kb is not None:
            sets.append("in_kb = ?")
            params.append(1 if in_kb else 0)
        if file_fingerprint is not None:
            sets.append("file_fingerprint = ?")
            params.append(file_fingerprint)
        params.append(suggestion_id)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE suggested_readings SET {', '.join(sets)} WHERE id = ?",
                    params,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
