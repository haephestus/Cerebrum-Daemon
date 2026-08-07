"""
cerebrum_core.engrams.storage.note_engram_repository.engrams
================================================================
Everything that reads or writes the `engrams` table itself, plus the
type-specific constructors (add_mcq/add_flashcard/add_short_question/
add_long_question) that take raw generator-output dicts. The actual
per-type column mapping lives in content_codecs.py — this file is about
orchestration (create the engrams row, hand off to the right codec, wire
mastery joins for reads), not about knowing every content table's shape.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from typing import Optional

from cerebrum_core.engrams.core.types import Engram, EngramType
from . import content_codecs
from ._base import _id


class EngramsMixin:
    # -----------------------------------------------------------------------
    # Reads
    # -----------------------------------------------------------------------

    def get_engram(self, engram_id: str) -> Optional[Engram]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM engrams WHERE id = ?", (engram_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_engram(conn, row)
        finally:
            conn.close()

    def get_note_engrams(
        self, note_id: str, user_id: str, state: Optional[str] = None
    ) -> list[Engram]:
        """All active engrams generated from a specific note OWNED BY
        user_id, optionally filtered by this user's mastery state.

        Joins through notes so a user can't fetch engrams for a note_id
        they don't own just by knowing/guessing the id. LEFT JOIN on
        engram_mastery so engrams with no mastery row yet for this user
        (never attempted) still show up — their mastery fields will just
        be NULL unless filtered out by an explicit state filter."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT e.* FROM engrams e
                JOIN notes n ON n.id = e.note_id
                LEFT JOIN engram_mastery em
                  ON em.engram_id = e.id AND em.user_id = ?
                WHERE e.note_id = ?
                  AND n.user_id = ?
                  AND e.is_active = 1
                  AND (? IS NULL OR em.state = ?)
                """,
                (user_id, note_id, user_id, state, state),
            ).fetchall()
            return [self._row_to_engram(conn, r) for r in rows]
        finally:
            conn.close()

    def get_all_engrams(self, user_id: str) -> list[Engram]:
        """All active engrams across every bubble/note OWNED BY user_id."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT e.* FROM engrams e
                JOIN notes n ON n.id = e.note_id
                WHERE n.user_id = ? AND e.is_active = 1
                """,
                (user_id,),
            ).fetchall()
            return [self._row_to_engram(conn, r) for r in rows]
        finally:
            conn.close()

    def get_bubble_engrams(
        self, bubble_id: str, user_id: str, state: Optional[str] = None
    ) -> list[Engram]:
        """All active engrams for every note OWNED BY user_id in a bubble,
        optionally filtered by this user's mastery state.

        LEFT JOIN so engrams with no mastery row yet for this user (never
        attempted) still show up when no state filter is applied."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT e.* FROM engrams e
                JOIN notes n ON n.id = e.note_id
                LEFT JOIN engram_mastery em
                  ON em.engram_id = e.id AND em.user_id = ?
                WHERE n.bubble_id = ?
                  AND n.user_id = ?
                  AND e.is_active = 1
                  AND (? IS NULL OR em.state = ?)
                """,
                (user_id, bubble_id, user_id, state, state),
            ).fetchall()
            return [self._row_to_engram(conn, r) for r in rows]
        finally:
            conn.close()

    def get_topic_engrams(self, user_id: str, topic: str) -> list[Engram]:
        """Matches MasteryRepository.get_topic_engrams(user_id, topic).
        Returns engrams for notes OWNED BY user_id whose topic matches —
        `user_id` now actually scopes the result (previously accepted for
        interface compatibility but unused, which meant any caller could
        pass any user_id and get every note's engrams for that topic).

        Groups on the topic ENTITY (notes.topic_id), resolved from the given
        name via its canonical slug — NOT notes.domain (the pipeline-tracking
        classification set by mark_analysed_inator). Returns [] if this user
        has no such topic."""
        conn = self._get_conn()
        try:
            topic_id = self._lookup_topic_id(conn, user_id, topic)
            if topic_id is None:
                return []
            rows = conn.execute(
                """
                SELECT e.* FROM engrams e
                JOIN notes n ON n.id = e.note_id
                WHERE n.topic_id = ? AND n.user_id = ? AND e.is_active = 1
                """,
                (topic_id, user_id),
            ).fetchall()
            return [self._row_to_engram(conn, r) for r in rows]
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Generic creation (already-built Engram dataclass)
    # -----------------------------------------------------------------------

    def create_engram(self, engram: Engram) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engrams (id, bubble_id, note_id, type, target_cognitive_level, tags, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    engram.id,
                    engram.bubble_id,
                    engram.note_id,
                    engram.type.value,
                    engram.target_cognitive_level,
                    json.dumps(engram.tags),
                    int(engram.is_active),
                ),
            )
            self._insert_typed_content(conn, engram.id, engram.type, asdict(engram.content))
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Type-specific engram creation — take the RAW dict(s) a generator
    # emits (matching engram_mcq_v1 / engram_flashcard_v1 /
    # engram_short_question_v1 / engram_long_question_v1 shapes) and write
    # both the parent `engrams` row and the type table in one call, so the
    # generator pipeline doesn't need to hand-build an Engram/*Content
    # dataclass first. Each returns the new engram_id.
    #
    # These are thin wrappers around create_engram's two-step insert
    # (engrams row + content codec); create_engram itself is still there
    # for callers that already have a fully-built Engram dataclass.
    # -----------------------------------------------------------------------

    def _create_engram_row(
        self,
        conn: sqlite3.Connection,
        note_id: str,
        bubble_id: str,
        etype: EngramType,
        target_cognitive_level: int,
        tags: Optional[list],
        engram_id: Optional[str],
    ) -> str:
        engram_id = engram_id or _id()
        conn.execute(
            """
            INSERT OR IGNORE INTO engrams
              (id, bubble_id, note_id, type, target_cognitive_level, tags, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                engram_id,
                bubble_id,
                note_id,
                etype.value,
                target_cognitive_level,
                json.dumps(tags or []),
                1,
            ),
        )
        return engram_id

    def add_mcq(
        self,
        note_id: str,
        bubble_id: str,
        data: dict,
        target_cognitive_level: int = 1,
        tags: Optional[list] = None,
        engram_id: Optional[str] = None,
    ) -> str:
        """Create an mcq engram from a raw engram_mcq_v1 output dict
        (one item from MCQ_SCHEMA's output array — keys: finding_index,
        question_number, stem, options, correct_option,
        correct_explanation, distractor_notes, severity)."""
        conn = self._get_conn()
        try:
            eid = self._create_engram_row(
                conn, note_id, bubble_id, EngramType.MCQ,
                target_cognitive_level, tags, engram_id,
            )
            self._insert_typed_content(conn, eid, EngramType.MCQ, data)
            conn.commit()
            return eid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_flashcard(
        self,
        note_id: str,
        bubble_id: str,
        data: dict,
        target_cognitive_level: int = 1,
        tags: Optional[list] = None,
        engram_id: Optional[str] = None,
    ) -> str:
        """Create a flashcard engram from a raw engram_flashcard_v1 output
        dict (one item from FLASHCARD_SCHEMA's output array — keys:
        finding_index, card_number, front, back, bridge_concept, severity,
        diagnostic_note)."""
        conn = self._get_conn()
        try:
            eid = self._create_engram_row(
                conn, note_id, bubble_id, EngramType.FLASHCARD,
                target_cognitive_level, tags, engram_id,
            )
            self._insert_typed_content(conn, eid, EngramType.FLASHCARD, data)
            conn.commit()
            return eid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_short_question(
        self,
        note_id: str,
        bubble_id: str,
        questions: list[dict],
        target_cognitive_level: int = 1,
        tags: Optional[list] = None,
        engram_id: Optional[str] = None,
    ) -> str:
        """Create a short_question engram from the FULL raw
        engram_short_question_v1 output array (SHORT_QUESTION_SCHEMA's
        output — a list of question dicts, each with keys: finding_index,
        question_number, level, stem, expected_answer, hint,
        context_anchored, severity). One short_question engram holds many
        short_question rows, one per question_index — mirrors the existing
        one-row-per-question_index shape."""
        conn = self._get_conn()
        try:
            eid = self._create_engram_row(
                conn, note_id, bubble_id, EngramType.SHORT_QUESTION,
                target_cognitive_level, tags, engram_id,
            )
            for q in questions:
                self._insert_typed_content(conn, eid, EngramType.SHORT_QUESTION, q)
            conn.commit()
            return eid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_long_question(
        self,
        note_id: str,
        bubble_id: str,
        data: dict,
        target_cognitive_level: int = 1,
        tags: Optional[list] = None,
        engram_id: Optional[str] = None,
    ) -> str:
        """Create a long_question (long_question) engram from a raw
        engram_long_question_v1 output dict (LONG_QUESTION_SCHEMA's output —
        keys: finding_index, question_stem, answer, parts (list of
        part/level/question/marks/mark_scheme/note), severity,
        total_marks). Writes one long_question_content row plus one
        long_question_parts row per part."""
        conn = self._get_conn()
        try:
            eid = self._create_engram_row(
                conn, note_id, bubble_id, EngramType.LONG_QUESTION,
                target_cognitive_level, tags, engram_id,
            )
            self._insert_typed_content(conn, eid, EngramType.LONG_QUESTION, data)
            conn.commit()
            return eid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Codec glue — kept as methods (not just module-level content_codecs
    # calls inline) so any existing subclass overriding these private hooks
    # keeps working; they just delegate to content_codecs now.
    # -----------------------------------------------------------------------

    def _insert_typed_content(
        self, conn: sqlite3.Connection, engram_id: str, etype: EngramType, data: dict
    ) -> None:
        content_codecs.insert_typed_content(conn, engram_id, etype, data)

    def _load_content(self, conn: sqlite3.Connection, engram_id: str, etype: EngramType):
        return content_codecs.load_content(conn, engram_id, etype)

    def _row_to_engram(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Engram:
        etype = EngramType(row["type"])
        return Engram(
            id=row["id"],
            bubble_id=row["bubble_id"],
            note_id=row["note_id"],
            type=etype,
            target_cognitive_level=int(row["target_cognitive_level"]),
            content=self._load_content(conn, row["id"], etype),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
