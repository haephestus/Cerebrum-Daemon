"""
cerebrum_core.engrams.storage.note_engram_repository.notes
==============================================================
Everything that reads or writes the `notes` table: the legacy
NoteRegisterInator registry API (kept at identical names/signatures for
drop-in compatibility) plus the newer content CRUD (create_note/get_note).

Both halves live in one mixin, deliberately not split further, because
they share one table's invariants (cached/analysed/topic/domain/subject
all live on the same row) — splitting them would mean two files
implicitly coordinating on the same schema instead of one place owning it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class NotesMixin:
    # -----------------------------------------------------------------------
    # Notes — registry-style methods (formerly NoteRegisterInator)
    #
    # Method names + signatures below were kept identical to the original
    # NoteRegisterInator for drop-in compatibility, EXCEPT register_inator,
    # which now takes a required `user_id` (notes need an owner — see
    # notes.user_id in schema.py). This is a deliberate break from the
    # "identical signature" promise elsewhere in this file: every existing
    # register_inator(note_id, bubble_id, filepath) call site needs updating
    # to register_inator(note_id, user_id, bubble_id, filepath).
    # -----------------------------------------------------------------------

    def register_inator(
        self, note_id: str, user_id: str, bubble_id: Optional[str], filepath: str
    ):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO notes (id, user_id, bubble_id, filepath)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        last_analysed = CURRENT_TIMESTAMP
                    """,
                    (note_id, user_id, bubble_id, filepath),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def mark_cached_inator(self, note_id: str):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE notes SET cached = 1, last_analysed = CURRENT_TIMESTAMP WHERE id = ?",
                    (note_id,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def mark_analysed_inator(
        self,
        note_id: str,
        domain: Optional[str] = "",
        subject: Optional[str] = "",
    ):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE notes
                    SET analysed = 1,
                        domain = COALESCE(?, domain),
                        subject = COALESCE(?, subject),
                        last_analysed = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (domain, subject, note_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def mark_analysis_status(
        self, note_id: str, status: str, error: Optional[str] = None
    ) -> None:
        VALID = {"not_started", "pending", "running", "done", "failed"}
        if status not in VALID:
            raise ValueError(f"Invalid analysis_status: {status}")
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE notes
                    SET analysis_status = ?, analysis_error = ?,
                        analysed = CASE WHEN ? = 'done' THEN 1 ELSE analysed END,
                        last_analysed = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status, error, status, note_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_analysis_status(self, note_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT analysis_status, analysis_error, last_analysed FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def fetch_uncached_notes_inator(self, user_id: Optional[str] = None):
        """If user_id is given, only that user's uncached notes are
        returned; otherwise every uncached note (e.g. for a background
        ingestion worker that runs across all users)."""
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute(
                    "SELECT id AS note_id, user_id, bubble_id, filepath FROM notes WHERE cached = 0 AND user_id = ?",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id AS note_id, user_id, bubble_id, filepath FROM notes WHERE cached = 0"
                ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def fetch_unanalysed_notes_inator(self, user_id: Optional[str] = None):
        """If user_id is given, only that user's unanalysed notes are
        returned; otherwise every unanalysed note across all users."""
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT id AS note_id, user_id, bubble_id, domain, subject, filepath
                    FROM notes WHERE analysed = 0 AND user_id = ?
                    """,
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id AS note_id, user_id, bubble_id, domain, subject, filepath
                    FROM notes WHERE analysed = 0
                    """
                ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def check_inator(self, note_id: str, field: str = "") -> bool:
        VALID_FIELDS = {"cached", "analysed"}
        conn = self._get_conn()
        try:
            if field:
                if field not in VALID_FIELDS:
                    raise ValueError("Invalid field requested")
                result = conn.execute(
                    f"SELECT {field} FROM notes WHERE id = ?", (note_id,)
                ).fetchone()
            else:
                result = conn.execute(
                    "SELECT 1 FROM notes WHERE id = ?", (note_id,)
                ).fetchone()
        finally:
            conn.close()
        return bool(result and (result[0] if field else True))

    def show_all_inator(self, user_id: Optional[str] = None):
        """If user_id is given, only that user's notes are returned;
        otherwise every note (admin/debug use)."""
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM notes WHERE user_id = ?", (user_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM notes").fetchall()
        finally:
            conn.close()

        # Original show_all_inator's dicts keyed the id column as "note_id"
        # (note_registry's own naming), not "id". Preserve that key name so
        # callers unpacking row["note_id"] etc. still work unchanged.
        result = []
        for r in rows:
            d = dict(r)
            d["note_id"] = d.pop("id")
            result.append(d)
        return result

    def remove_inator(self, note_id: str, filepath: str):
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                if cur.rowcount == 0:
                    raise FileNotFoundError("Note registry entry not found")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        path = Path(filepath)
        if path.exists():
            path.unlink()

    def reset_inator(self, status: str, note_id: Optional[str] = None):
        VALID_COLUMNS = {"cached", "analysed"}
        if status not in VALID_COLUMNS:
            raise ValueError("Invalid status field")

        with self._lock:
            conn = self._get_conn()
            try:
                if note_id:
                    cur = conn.execute(
                        f"UPDATE notes SET {status} = 0 WHERE id = ?", (note_id,)
                    )
                else:
                    cur = conn.execute(f"UPDATE notes SET {status} = 0")
                conn.commit()
                count = cur.rowcount
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return count

    # -----------------------------------------------------------------------
    # Notes — content CRUD (formerly implicit in SQLiteRepository; it never
    # actually had a create_note, per its own TODO — added here now that
    # notes and note-tracking share a table, so engram_generator_inator can
    # create the note a generated Engram's note_id points at).
    # -----------------------------------------------------------------------

    def create_note(
        self,
        note_id: str,
        user_id: str,
        content: str,
        bubble_id: Optional[str] = None,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        tags: Optional[list] = None,
    ):
        # Resolve the topic to its entity (id + canonical name) BEFORE taking
        # the write lock — resolve_topic takes the same non-reentrant lock, so
        # calling it inside the block below would deadlock. This both dedupes
        # spelling variants (via slug) and gives the note a stable topic_id.
        topic_id: Optional[str] = None
        if topic:
            resolved = self.resolve_topic(user_id, topic)
            if resolved:
                topic_id = resolved["id"]
                topic = resolved["name"]  # canonical display name
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO notes (id, user_id, bubble_id, domain, subject, topic, topic_id, content, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        tags = excluded.tags,
                        topic = COALESCE(excluded.topic, notes.topic),
                        topic_id = COALESCE(excluded.topic_id, notes.topic_id),
                        version = version + 1,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        note_id,
                        user_id,
                        bubble_id,
                        domain,
                        subject,
                        topic,
                        topic_id,
                        content,
                        json.dumps(tags or []),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_note(self, note_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def assign_note_topic(self, note_id: str, topic: str) -> Optional[dict]:
        """Set a note's topic from the analysis result.

        This is where topics actually enter the system: a note is registered
        (register_inator) with no topic; analysis later determines it
        (note_overview.topic); this persists it via the topic entity so
        mastery/engrams/planner can group on it. Resolves the name to a
        topic_id (deduping spellings) and updates notes.topic + notes.topic_id.
        Returns the resolved {id, name, slug}, or None if the note doesn't
        exist or the topic is blank. Resolves BEFORE taking the write lock,
        since resolve_topic takes the same non-reentrant lock.
        """
        note = self.get_note(note_id)
        if not note:
            return None
        if not topic:
            return None
        resolved = self.resolve_topic(note["user_id"], topic)
        if not resolved:
            return None
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE notes SET topic = ?, topic_id = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (resolved["name"], resolved["id"], note_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return resolved
