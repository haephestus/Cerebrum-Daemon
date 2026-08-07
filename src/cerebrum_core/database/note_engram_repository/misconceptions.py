"""
cerebrum_core.engrams.storage.note_engram_repository.misconceptions
=======================================================================
Tracks recurring misconceptions per user/engram/concept. Single method
today; kept as its own mixin rather than folded into mastery.py because
it's conceptually distinct (a log of specific misunderstandings, not a
mastery score) even though both are attempt-derived signals.
"""

from __future__ import annotations

from ._base import _id


class MisconceptionsMixin:
    def upsert_misconception(
        self, user_id: str, engram_id: str, concept: str, description: str
    ) -> None:
        conn = self._get_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM misconceptions WHERE user_id=? AND engram_id=? AND concept=?",
                (user_id, engram_id, concept),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE misconceptions SET occurrences=occurrences+1, last_seen=datetime('now'), description=? WHERE id=?",
                    (description, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO misconceptions (id, user_id, engram_id, concept, description) VALUES (?, ?, ?, ?, ?)",
                    (_id(), user_id, engram_id, concept, description),
                )
            conn.commit()
        finally:
            conn.close()
