"""
cerebrum_core.database.note_engram_repository.topics
=====================================================
The topics ENTITY: read/write for the `topics` table plus the shared
name->id resolution the mastery/engram/attempt mixins use so all topic
grouping keys off a stable id instead of a raw string.

  * resolve_topic  — write-path upsert: name -> canonical topic row
                     (creates it on first sight, keyed by slug).
  * _lookup_topic_id — read-path resolution used inside other mixins'
                     queries: name -> existing topic_id, or None.
  * rename_topic   — change a topic's display name safely: the id stays,
                     the name and its denormalised copies on notes /
                     topic_mastery are updated in one transaction, so a
                     rename never fragments mastery.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from cerebrum_core.utils.topic_inator import normalize_topic, topic_slug

from ._base import _id, _now


class TopicsMixin:
    # -- resolution -------------------------------------------------------
    def _lookup_topic_id(
        self, conn: sqlite3.Connection, user_id: str, name: str
    ) -> Optional[str]:
        """Read-only: the topic_id for `name` under `user_id`, matched by
        canonical slug, or None if this user has no such topic yet. Takes an
        open connection so callers can use it inside their own query without
        opening a second one."""
        slug = topic_slug(name)
        if not slug:
            return None
        row = conn.execute(
            "SELECT id FROM topics WHERE user_id = ? AND slug = ?", (user_id, slug)
        ).fetchone()
        return row["id"] if row else None

    def resolve_topic(self, user_id: str, name: str) -> Optional[dict]:
        """Upsert a topic by canonical slug and return {id, name, slug}.
        Returns None for an empty/blank name (nothing to group on)."""
        slug = topic_slug(name)
        if not slug:
            return None
        canonical_name = normalize_topic(name)
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT id, name, slug FROM topics WHERE user_id = ? AND slug = ?",
                    (user_id, slug),
                ).fetchone()
                if row:
                    return dict(row)
                topic_id = _id()
                conn.execute(
                    "INSERT INTO topics (id, user_id, slug, name) VALUES (?, ?, ?, ?)",
                    (topic_id, user_id, slug, canonical_name),
                )
                conn.commit()
                return {"id": topic_id, "name": canonical_name, "slug": slug}
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # -- reads ------------------------------------------------------------
    def get_topic(self, user_id: str, topic_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM topics WHERE user_id = ? AND id = ?",
                (user_id, topic_id),
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def get_topic_id_by_name(self, user_id: str, name: str) -> Optional[str]:
        conn = self._get_conn()
        try:
            return self._lookup_topic_id(conn, user_id, name)
        finally:
            conn.close()

    def list_topics(self, user_id: str) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM topics WHERE user_id = ? ORDER BY name", (user_id,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # -- rename (id-stable) -----------------------------------------------
    def rename_topic(self, user_id: str, topic_id: str, new_name: str) -> Optional[dict]:
        """Rename a topic's display name. The id is unchanged, so mastery
        stays intact; the new canonical name is propagated to the
        denormalised notes.topic / topic_mastery.topic copies in the same
        transaction. Returns the updated {id, name, slug}, or None if the
        topic doesn't exist. Raises ValueError if new_name is blank or would
        collide with another existing topic's slug for this user."""
        new_slug = topic_slug(new_name)
        if not new_slug:
            raise ValueError("new_name is empty after normalisation")
        canonical_name = normalize_topic(new_name)
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM topics WHERE user_id = ? AND id = ?",
                (user_id, topic_id),
            ).fetchone()
            if not existing:
                return None
            clash = conn.execute(
                "SELECT id FROM topics WHERE user_id = ? AND slug = ? AND id <> ?",
                (user_id, new_slug, topic_id),
            ).fetchone()
            if clash:
                raise ValueError(
                    f"rename would collide with existing topic {clash['id']} "
                    f"(slug {new_slug!r}); merge is not supported here"
                )
            conn.execute(
                "UPDATE topics SET name = ?, slug = ?, updated_at = ? WHERE id = ?",
                (canonical_name, new_slug, _now(), topic_id),
            )
            conn.execute(
                "UPDATE notes SET topic = ? WHERE topic_id = ?",
                (canonical_name, topic_id),
            )
            conn.execute(
                "UPDATE topic_mastery SET topic = ? WHERE topic_id = ?",
                (canonical_name, topic_id),
            )
        return {"id": topic_id, "name": canonical_name, "slug": new_slug}
