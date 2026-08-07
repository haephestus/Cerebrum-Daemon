"""
cerebrum_core.engrams.storage.note_engram_repository.content_codecs
=======================================================================
Type-specific engram content (de)serialization: one write_* / read_*
function pair per engram type (mcq, flashcard, short_question,
long_question), plus insert_typed_content/load_content dispatchers.

This used to be two ~80-line if/elif ladders (_insert_typed_content,
_load_content) living on the repository class itself. Pulled out here so
each type's read/write shape is a small, independently testable unit
instead of one branch in a growing dispatcher — and so engrams.py stays
about orchestration (create the engrams row, call the right codec), not
about knowing every content table's columns.
"""

from __future__ import annotations

import sqlite3

from ...engrams.core.types import (EngramType, FlashcardContent, LongQuestionContent,
                          LongQuestionPart, MCQContent, QuizContent,
                          QuizQuestion)
from ._base import _id

# ---------------------------------------------------------------------------
# MCQ — matches engram_mcq_v1 output.
# ---------------------------------------------------------------------------


def write_mcq_content(conn: sqlite3.Connection, engram_id: str, data: dict) -> None:
    opts = data.get("options", {})
    distractor = data.get("distractor_notes", {}) or {}
    conn.execute(
        """
        INSERT OR REPLACE INTO mcq_content
          (engram_id, finding_index, question_number, question,
           option_a, option_b, option_c, option_d, correct_option,
           explanation, severity,
           distractor_misconception_option, distractor_confused_link_option)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            engram_id,
            data.get("finding_index"),
            data.get("question_number"),
            data["stem"],
            opts.get("A", ""),
            opts.get("B", ""),
            opts.get("C", ""),
            opts.get("D", ""),
            data["correct_option"],
            data["correct_explanation"],
            data.get("severity"),
            distractor.get("misconception_option"),
            distractor.get("confused_link_option"),
        ),
    )


def read_mcq_content(conn: sqlite3.Connection, engram_id: str) -> MCQContent:
    r = conn.execute(
        "SELECT * FROM mcq_content WHERE engram_id = ?", (engram_id,)
    ).fetchone()
    return MCQContent(
        finding_index=r["finding_index"],
        question_number=r["question_number"],
        stem=r["question"],
        options={
            "A": r["option_a"],
            "B": r["option_b"],
            "C": r["option_c"],
            "D": r["option_d"],
        },
        correct_option=r["correct_option"],
        explanation=r["explanation"],
        severity=r["severity"],
        distractor_notes={
            k: v
            for k, v in {
                "misconception_option": r["distractor_misconception_option"],
                "confused_link_option": r["distractor_confused_link_option"],
            }.items()
            if v is not None
        },
    )


# ---------------------------------------------------------------------------
# Flashcard — matches engram_flashcard_v1 output.
# ---------------------------------------------------------------------------


def write_flashcard_content(
    conn: sqlite3.Connection, engram_id: str, data: dict
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO flashcard_content
          (engram_id, finding_index, card_number, front, back,
           bridge_concept, severity, diagnostic_note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            engram_id,
            data.get("finding_index"),
            data.get("card_number"),
            data["front"],
            data["back"],
            data.get("bridge_concept"),
            data.get("severity"),
            data.get("diagnostic_note"),
        ),
    )


def read_flashcard_content(
    conn: sqlite3.Connection, engram_id: str
) -> FlashcardContent:
    r = conn.execute(
        "SELECT * FROM flashcard_content WHERE engram_id = ?", (engram_id,)
    ).fetchone()
    return FlashcardContent(
        finding_index=r["finding_index"],
        card_number=r["card_number"],
        front=r["front"],
        back=r["back"],
        bridge_concept=r["bridge_concept"],
        severity=r["severity"],
        diagnostic_note=r["diagnostic_note"],
    )


# ---------------------------------------------------------------------------
# Short question — matches engram_short_question_v1 output. One engram
# holds many rows (one per question_index), so write_ is called once per
# question by the caller, while read_ reassembles the whole QuizContent.
# ---------------------------------------------------------------------------


def write_short_question(conn: sqlite3.Connection, engram_id: str, data: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO short_question
          (id, engram_id, finding_index, question_index, level,
           stem, expected_answer, hint, context_anchored, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _id(),
            engram_id,
            data.get("finding_index"),
            data["question_number"],
            data["level"],
            data["stem"],
            data["expected_answer"],
            data.get("hint"),
            int(bool(data.get("context_anchored", False))),
            data.get("severity"),
        ),
    )


def read_short_question_content(
    conn: sqlite3.Connection, engram_id: str
) -> QuizContent:
    rows = conn.execute(
        "SELECT * FROM short_question WHERE engram_id = ? ORDER BY question_index",
        (engram_id,),
    ).fetchall()
    return QuizContent(
        questions=[
            QuizQuestion(
                finding_index=r["finding_index"],
                question_number=r["question_index"],
                level=r["level"],
                stem=r["stem"],
                expected_answer=r["expected_answer"],
                hint=r["hint"],
                context_anchored=bool(r["context_anchored"]),
                severity=r["severity"],
            )
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# Long question — matches engram_long_question_v1 output. One
# long_question_content parent row plus one long_question_parts row per
# scaffolded part (part a/b/c/...), each independently marked.
# ---------------------------------------------------------------------------


def write_long_question_content(
    conn: sqlite3.Connection, engram_id: str, data: dict
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO long_question_content
          (engram_id, finding_index, question_stem, answer, severity, total_marks)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            engram_id,
            data.get("finding_index"),
            data["question_stem"],
            data.get("answer"),
            data.get("severity"),
            data.get("total_marks"),
        ),
    )
    conn.execute("DELETE FROM long_question_parts WHERE engram_id = ?", (engram_id,))
    for part in data.get("parts", []):
        conn.execute(
            """
            INSERT INTO long_question_parts
              (id, engram_id, part, level, question, marks, mark_scheme, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _id(),
                engram_id,
                part["part"],
                part["level"],
                part["question"],
                part["marks"],
                part["mark_scheme"],
                part.get("note"),
            ),
        )


def read_long_question_content(
    conn: sqlite3.Connection, engram_id: str
) -> LongQuestionContent:
    r = conn.execute(
        "SELECT * FROM long_question_content WHERE engram_id = ?", (engram_id,)
    ).fetchone()
    parts = conn.execute(
        "SELECT * FROM long_question_parts WHERE engram_id = ? ORDER BY part",
        (engram_id,),
    ).fetchall()
    return LongQuestionContent(
        finding_index=r["finding_index"],
        question_stem=r["question_stem"],
        answer=r["answer"],
        severity=r["severity"],
        total_marks=r["total_marks"],
        parts=[
            LongQuestionPart(
                part=p["part"],
                level=p["level"],
                question=p["question"],
                marks=p["marks"],
                mark_scheme=p["mark_scheme"],
                note=p["note"],
            )
            for p in parts
        ],
    )


# ---------------------------------------------------------------------------
# Dispatchers — used by engrams.py so it doesn't need its own if/elif ladder
# ---------------------------------------------------------------------------

_WRITERS = {
    EngramType.MCQ: write_mcq_content,
    EngramType.FLASHCARD: write_flashcard_content,
    EngramType.SHORT_QUESTION: write_short_question,
    EngramType.LONG_QUESTION: write_long_question_content,
}

_READERS = {
    EngramType.MCQ: read_mcq_content,
    EngramType.FLASHCARD: read_flashcard_content,
    EngramType.SHORT_QUESTION: read_short_question_content,
    EngramType.LONG_QUESTION: read_long_question_content,
}


def insert_typed_content(
    conn: sqlite3.Connection, engram_id: str, etype: EngramType, data: dict
) -> None:
    try:
        writer = _WRITERS[etype]
    except KeyError:
        raise ValueError(f"Unknown engram type: {etype}") from None
    writer(conn, engram_id, data)


def load_content(conn: sqlite3.Connection, engram_id: str, etype: EngramType):
    try:
        reader = _READERS[etype]
    except KeyError:
        raise ValueError(f"Unknown engram type: {etype}") from None
    return reader(conn, engram_id)
