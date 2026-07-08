"""
cerebrum_core.engrams.storage.note_engram_repository
======================================================
Merged replacement for the old pair of:
  - cerebrum_core.utils.registry.note_registry.NoteRegisterInator
  - cerebrum_core.engrams.storage.sqlite_repository.SQLiteRepository

Why merged: both classes owned a table describing "a note" (note_registry.note_id
vs notes.id), living in two separate .db files with no way to JOIN them. This
repository owns a single `notes` table used both for ingestion-pipeline tracking
(cached/analysed/filepath) and as the note content engrams hang off of.

Schema change vs the old `notes` table: `title`, `subtopic`, `source`, and
`embedding_id` are dropped. `topic` is kept — it's what the note is about,
and is genuinely distinct from `domain`/`subject` (the pipeline-tracking
classification that note_registry's mark_analysed_inator sets). Engrams and
mastery group on `topic`, never on `domain`.

CONTENT-TABLE SHAPE (2024 revision): mcq_content, flashcard_content,
short_answer_questions, and long_question_content/long_question_parts below are
shaped to match the ACTUAL generator output schemas (engram_mcq_v1,
engram_flashcard_v1, engram_short_answer_v1, engram_long_answer_v1) rather than an
earlier, incorrect assumption that all four content types were MCQ-like
with a rubric. See inline comments on each table for what changed and why.

Drop-in compatibility: every method that used to live on NoteRegisterInator
(register_inator, mark_cached_inator, mark_analysed_inator,
fetch_uncached_notes_inator, fetch_unanalysed_notes_inator, check_inator,
show_all_inator, remove_inator, reset_inator) keeps its exact name and
signature here, so existing call sites only need to change the import:

    - from cerebrum_core.utils.registry.note_registry_inator import NoteRegisterInator
    + from cerebrum_core.engrams.storage.note_engram_repository import NoteEngramRepository as NoteRegisterInator

The mastery/engram side implements the current MasteryRepository ABC
(mastery_service.py): get_topic_masteries, upsert_topic_mastery,
get_topic_mastery, get_topic_engrams(user_id, topic).

Usage (existing connection, e.g. in-memory DB for tests):
    import sqlite3
    from cerebrum_core.engrams.storage.note_engram_repository import NoteEngramRepository

    db   = sqlite3.connect(":memory:")
    repo = NoteEngramRepository(db)  # schema created automatically

Usage (production, resolves the on-disk registry path for you):
    from cerebrum_core.engrams.storage.note_engram_repository import NoteEngramRepository

    repo = NoteEngramRepository.open()  # or NoteEngramRepository.open("registry/notes.db")
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from cerebrum_core.utils.file_util_inator import CerebrumPaths

from ..core.mastery_service import MasteryRepository
from ..core.types import (
    DimensionScores,
    Engram,
    EngramAttempt,
    EngramMastery,
    EngramType,
    FlashcardRating,
    FlashcardResponse,
    GraderType,
    LongQuestionResponse,
    MasteryState,
    MCQResponse,
    QuizResponse,
    TopicMastery,
)


def _now() -> str:
    return datetime.utcnow().isoformat()


def _id() -> str:
    return uuid.uuid4().hex


_DEFAULT_DB_PATH = "registry/note_registry.db"

# ---------------------------------------------------------------------------
# Schema — single source of truth for both note tracking and engrams/mastery.
# All CREATE TABLE statements use IF NOT EXISTS, so re-running against an
# existing DB is safe.
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS notes (
  id            TEXT PRIMARY KEY,     -- was note_registry.note_id
  bubble_id     TEXT,
  domain        TEXT,   -- pipeline tracking classification, set by mark_analysed_inator
  subject       TEXT,   -- pipeline tracking classification, set by mark_analysed_inator
  topic         TEXT,   -- what the note is ABOUT; engrams/mastery group on this, not domain
  cached        INTEGER NOT NULL DEFAULT 0,
  analysed      INTEGER NOT NULL DEFAULT 0,
  filepath      TEXT,
  content       TEXT NOT NULL DEFAULT '',
  tags          TEXT NOT NULL DEFAULT '[]',
  version       INTEGER NOT NULL DEFAULT 1,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
  last_analysed TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notes_domain  ON notes(domain);
CREATE INDEX IF NOT EXISTS idx_notes_topic   ON notes(topic);
CREATE INDEX IF NOT EXISTS idx_notes_tags    ON notes(tags);
CREATE INDEX IF NOT EXISTS idx_notes_cached  ON notes(cached);
CREATE INDEX IF NOT EXISTS idx_notes_analysed ON notes(analysed);

CREATE TABLE IF NOT EXISTS engrams (
  id              TEXT PRIMARY KEY,
  note_id         TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  type            TEXT NOT NULL CHECK(type IN ('mcq','flashcard','short_answer','long_question')),
  cognitive_level INTEGER NOT NULL DEFAULT 1 CHECK(cognitive_level BETWEEN 1 AND 7),
  tags            TEXT NOT NULL DEFAULT '[]',
  is_active       INTEGER NOT NULL DEFAULT 1,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_engrams_note_id         ON engrams(note_id);
CREATE INDEX IF NOT EXISTS idx_engrams_type            ON engrams(type);
CREATE INDEX IF NOT EXISTS idx_engrams_cognitive_level ON engrams(cognitive_level);

-- ---------------------------------------------------------------------
-- mcq_content — matches engram_mcq_v1 output.
--
-- Changed vs previous revision:
--   * rationale_a/b/c/d DROPPED. The generator never produces per-option
--     rationale text — `distractor_notes` only names WHICH option plays
--     which trap role. Those two are genuinely different data
--     (explanatory text per option vs. metadata about the options), so
--     renaming rationale_a->something wouldn't have been honest either.
--   * distractor_misconception_option / distractor_confused_link_option
--     ADDED — these store the option letter (A-D) that
--     distractor_notes.misconception_option / .confused_link_option
--     names, which is what the generator actually emits.
--   * finding_index, question_number, severity ADDED — present on every
--     generated card and previously silently dropped on write.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mcq_content (
  engram_id      TEXT PRIMARY KEY REFERENCES engrams(id) ON DELETE CASCADE,
  finding_index  INTEGER,
  question_number INTEGER,
  question       TEXT NOT NULL,
  option_a       TEXT NOT NULL,
  option_b       TEXT NOT NULL,
  option_c       TEXT NOT NULL,
  option_d       TEXT NOT NULL,
  correct_option TEXT NOT NULL CHECK(correct_option IN ('A','B','C','D')),
  explanation    TEXT NOT NULL,
  severity       TEXT,
  distractor_misconception_option TEXT
    CHECK(distractor_misconception_option IN ('A','B','C','D') OR distractor_misconception_option IS NULL),
  distractor_confused_link_option TEXT
    CHECK(distractor_confused_link_option IN ('A','B','C','D') OR distractor_confused_link_option IS NULL)
);

-- ---------------------------------------------------------------------
-- flashcard_content — matches engram_flashcard_v1 output.
--
-- Changed vs previous revision:
--   * hint, mnemonic DROPPED. The generator never emits these — they
--     were dead columns that always wrote NULL. Reintroduce them only
--     if some other write path actually populates them.
--   * bridge_concept, severity, diagnostic_note, finding_index,
--     card_number ADDED — present on every generated card and
--     previously silently dropped on write.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS flashcard_content (
  engram_id       TEXT PRIMARY KEY REFERENCES engrams(id) ON DELETE CASCADE,
  finding_index   INTEGER,
  card_number     INTEGER,
  front           TEXT NOT NULL,
  back            TEXT NOT NULL,
  bridge_concept  TEXT,
  severity        TEXT,
  diagnostic_note TEXT
);

-- ---------------------------------------------------------------------
-- short_answer_questions — matches engram_short_answer_v1 output.
--
-- Changed vs previous revision: this was MCQ-shaped (option_a-d,
-- correct_option), but the generator produces OPEN-RESPONSE
-- recall/explain/apply questions with a free-text expected_answer to be
-- graded (by an LLM grader), not a lettered choice. option_a-d and
-- correct_option are DROPPED; level, stem, expected_answer, hint,
-- context_anchored, severity, finding_index ADDED.
--
-- NOTE: this makes short_answer_responses (which stores selected_option /
-- correct_option) stale for this engram type — that table assumed
-- lettered answers too. It needs its own follow-up pass once the
-- grading path for open-response short_answer answers is decided; left
-- untouched here since that's a response-side, not content-side, change.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS short_answer_questions (
  id               TEXT PRIMARY KEY,
  engram_id        TEXT NOT NULL REFERENCES engrams(id) ON DELETE CASCADE,
  finding_index    INTEGER,
  question_index   INTEGER NOT NULL,
  level            TEXT NOT NULL CHECK(level IN ('recall','understand','apply','synthesise','evaluate','doctoral')),
  stem             TEXT NOT NULL,
  expected_answer  TEXT NOT NULL,
  hint             TEXT,
  context_anchored INTEGER NOT NULL DEFAULT 0,
  severity         TEXT,
  UNIQUE(engram_id, question_index)
);

CREATE INDEX IF NOT EXISTS idx_short_answer_questions_engram ON short_answer_questions(engram_id, question_index);

-- ---------------------------------------------------------------------
-- long_question_content / long_question_parts — matches
-- engram_long_answer_v1 output.
--
-- Changed vs previous revision: the old table assumed one flat question
-- with one holistic multi-dimension rubric (rubric_accuracy, ...,
-- rubric_awareness_of_limits). The generator instead produces ONE
-- scaffolded multi-part question (part a/b/c/..., each independently
-- marked with its own `marks` + `mark_scheme`) — a one-to-many shape,
-- not one-to-one, and there is no holistic rubric at all in the actual
-- output. rubric_* columns DROPPED from the parent; a new child table
-- long_question_parts holds one row per part, mirroring how
-- short_answer_questions already does one row per question_index.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS long_question_content (
  engram_id     TEXT PRIMARY KEY REFERENCES engrams(id) ON DELETE CASCADE,
  finding_index INTEGER,
  question_stem TEXT NOT NULL,
  answer        TEXT,
  severity      TEXT,
  total_marks   INTEGER
);

CREATE TABLE IF NOT EXISTS long_question_parts (
  id          TEXT PRIMARY KEY,
  engram_id   TEXT NOT NULL REFERENCES engrams(id) ON DELETE CASCADE,
  part        TEXT NOT NULL,
  level       TEXT NOT NULL CHECK(level IN ('recall','understand','apply','synthesise','evaluate','doctoral')),
  question    TEXT NOT NULL,
  marks       INTEGER NOT NULL,
  mark_scheme TEXT NOT NULL,
  note        TEXT,
  UNIQUE(engram_id, part)
);

CREATE INDEX IF NOT EXISTS idx_long_question_parts_engram ON long_question_parts(engram_id);

CREATE TABLE IF NOT EXISTS users (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  email      TEXT UNIQUE NOT NULL,
  settings   TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS engram_attempts (
  id               TEXT PRIMARY KEY,
  engram_id        TEXT NOT NULL REFERENCES engrams(id),
  user_id          TEXT NOT NULL REFERENCES users(id),
  attempted_at     TEXT NOT NULL DEFAULT (datetime('now')),
  score            REAL,
  grader           TEXT NOT NULL DEFAULT 'pending'
                   CHECK(grader IN ('pending','auto','ai','human')),
  time_spent_ms    INTEGER,
  note_version     INTEGER,
  cognitive_level  INTEGER NOT NULL,
  context_snapshot TEXT
);

CREATE INDEX IF NOT EXISTS idx_attempts_engram_user ON engram_attempts(engram_id, user_id);
CREATE INDEX IF NOT EXISTS idx_attempts_user_time   ON engram_attempts(user_id, attempted_at);
CREATE INDEX IF NOT EXISTS idx_attempts_grader      ON engram_attempts(grader);

CREATE TABLE IF NOT EXISTS mcq_responses (
  attempt_id      TEXT PRIMARY KEY REFERENCES engram_attempts(id),
  selected_option TEXT NOT NULL,
  correct_option  TEXT NOT NULL,
  is_correct      INTEGER NOT NULL,
  distractor_key  TEXT
);

CREATE TABLE IF NOT EXISTS flashcard_responses (
  attempt_id      TEXT PRIMARY KEY REFERENCES engram_attempts(id),
  self_rating     TEXT NOT NULL CHECK(self_rating IN ('again','hard','good','easy')),
  time_to_flip_ms INTEGER
);

-- NOTE (see short_answer_questions comment above): selected_option/correct_option
-- here still assume a lettered answer. Left as-is pending a decision on
-- how open-response short_answer answers get graded/recorded.
CREATE TABLE IF NOT EXISTS short_answer_responses (
  id              TEXT PRIMARY KEY,
  attempt_id      TEXT NOT NULL REFERENCES engram_attempts(id),
  question_index  INTEGER NOT NULL,
  selected_option TEXT NOT NULL,
  correct_option  TEXT NOT NULL,
  is_correct      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS long_question_responses (
  attempt_id            TEXT PRIMARY KEY REFERENCES engram_attempts(id),
  raw_answer            TEXT NOT NULL,
  word_count            INTEGER,
  ai_feedback           TEXT,
  concepts_demonstrated TEXT,
  concepts_missed       TEXT,
  misconceptions        TEXT,
  dimension_scores      TEXT,
  level_demonstrated    INTEGER,
  regression_detected   INTEGER DEFAULT 0,
  vector_id             TEXT,
  graded_at             TEXT
);

CREATE INDEX IF NOT EXISTS idx_lqr_vector ON long_question_responses(vector_id);

CREATE TABLE IF NOT EXISTS engram_mastery (
  id                  TEXT PRIMARY KEY,
  engram_id           TEXT NOT NULL REFERENCES engrams(id),
  user_id             TEXT NOT NULL REFERENCES users(id),
  state               TEXT NOT NULL DEFAULT 'new'
                      CHECK(state IN ('new','learning','review','mastered','lapsed','suspended')),
  current_score       REAL NOT NULL DEFAULT 0.0,
  stability           REAL NOT NULL DEFAULT 0.0,
  interval_days       REAL NOT NULL DEFAULT 1.0,
  next_due_at         TEXT NOT NULL DEFAULT (datetime('now')),
  last_attempted_at   TEXT,
  attempt_count       INTEGER NOT NULL DEFAULT 0,
  lapse_count         INTEGER NOT NULL DEFAULT 0,
  consecutive_correct INTEGER NOT NULL DEFAULT 0,
  current_level       INTEGER NOT NULL DEFAULT 1,
  score_accuracy            REAL DEFAULT 0.0,
  score_depth               REAL DEFAULT 0.0,
  score_reasoning           REAL DEFAULT 0.0,
  score_connections         REAL DEFAULT 0.0,
  score_originality         REAL DEFAULT 0.0,
  score_precision           REAL DEFAULT 0.0,
  score_awareness_of_limits REAL DEFAULT 0.0,
  updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(engram_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_mastery_user_due    ON engram_mastery(user_id, next_due_at);
CREATE INDEX IF NOT EXISTS idx_mastery_user_state  ON engram_mastery(user_id, state);
CREATE INDEX IF NOT EXISTS idx_mastery_engram_user ON engram_mastery(engram_id, user_id);

-- `topic` here is genuinely distinct from notes.domain: domain is the
-- pipeline-tracking classification set by mark_analysed_inator; topic is
-- what the note is about, and is what engrams/mastery actually group on.
-- Matches TopicMastery.topic in cerebrum_core.engrams.core.types.
CREATE TABLE IF NOT EXISTS topic_mastery (
  id               TEXT PRIMARY KEY,
  user_id          TEXT NOT NULL REFERENCES users(id),
  topic            TEXT NOT NULL,
  factual_score    REAL NOT NULL DEFAULT 0.0,
  applied_score    REAL NOT NULL DEFAULT 0.0,
  conceptual_score REAL NOT NULL DEFAULT 0.0,
  doctoral_score   REAL NOT NULL DEFAULT 0.0,
  overall_score    REAL NOT NULL DEFAULT 0.0,
  engram_count     INTEGER NOT NULL DEFAULT 0,
  lapsed_count     INTEGER NOT NULL DEFAULT 0,
  updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(user_id, topic)
);

CREATE INDEX IF NOT EXISTS idx_topic_mastery_user ON topic_mastery(user_id);

CREATE TABLE IF NOT EXISTS misconceptions (
  id          TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL REFERENCES users(id),
  engram_id   TEXT NOT NULL REFERENCES engrams(id),
  concept     TEXT NOT NULL,
  description TEXT NOT NULL,
  first_seen  TEXT NOT NULL DEFAULT (datetime('now')),
  last_seen   TEXT NOT NULL DEFAULT (datetime('now')),
  occurrences INTEGER NOT NULL DEFAULT 1,
  resolved    INTEGER NOT NULL DEFAULT 0,
  resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_misconceptions_user     ON misconceptions(user_id);
CREATE INDEX IF NOT EXISTS idx_misconceptions_resolved ON misconceptions(user_id, resolved);

CREATE TABLE IF NOT EXISTS grading_jobs (
  id           TEXT PRIMARY KEY,
  attempt_id   TEXT NOT NULL REFERENCES engram_attempts(id),
  status       TEXT NOT NULL DEFAULT 'pending'
               CHECK(status IN ('pending','processing','done','failed')),
  priority     INTEGER NOT NULL DEFAULT 5,
  attempts     INTEGER NOT NULL DEFAULT 0,
  error        TEXT,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  started_at   TEXT,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_grading_jobs_status ON grading_jobs(status, priority);

CREATE TABLE IF NOT EXISTS engram_generation_queue (
  id           TEXT PRIMARY KEY,
  note_id      TEXT NOT NULL REFERENCES notes(id),
  user_id      TEXT NOT NULL REFERENCES users(id),
  trigger      TEXT NOT NULL,
  trigger_ref  TEXT,
  target_congnitive_level INTEGER NOT NULL,
  target_type  TEXT NOT NULL,
  instructions TEXT,
  status       TEXT NOT NULL DEFAULT 'pending',
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class NoteEngramRepository(MasteryRepository):
    """
    Owns notes (ingestion tracking + content), engrams, attempts, mastery,
    misconceptions, and job queues in a single SQLite file.

    Thread-safety: unlike the old SQLiteRepository (one held connection),
    this opens a short-lived connection per call, guarded by a lock for
    writes — matching NoteRegisterInator's pattern, since this is what
    gets used from a threaded FastAPI app via app.state.
    """

    _lock = threading.Lock()

    def __init__(
        self, db_path: Union[str, Path, None] = None, ensure_schema: bool = True
    ):
        self.DB_PATH = CerebrumPaths().kb_root_dir() / (db_path or _DEFAULT_DB_PATH)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ensure_schema:
            self._ensure_schema()

    @classmethod
    def open(cls, db_path: Union[str, Path, None] = None) -> "NoteEngramRepository":
        return cls(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.DB_PATH, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -----------------------------------------------------------------------
    # Schema setup + migration
    # -----------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def _columns_of(self, conn: sqlite3.Connection, table: str) -> list[str]:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

    def _insert_typed_content(
        self, conn: sqlite3.Connection, engram_id: str, etype: "EngramType", data: dict
    ) -> None:
        if etype == EngramType.MCQ:
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
        elif etype == EngramType.FLASHCARD:
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
        elif etype == EngramType.QUIZ:
            # `data` here is a single generated short_answer question (the caller
            # loops over the generator's output array and calls this once
            # per question), matching how short_answer_questions already stores one
            # row per question_index.
            conn.execute(
                """
                INSERT OR REPLACE INTO short_answer_questions
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
        elif etype == EngramType.LONG_QUESTION:
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
            conn.execute(
                "DELETE FROM long_question_parts WHERE engram_id = ?", (engram_id,)
            )
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
        else:
            raise ValueError(f"Unknown engram type: {etype}")

    # -----------------------------------------------------------------------
    # Notes — registry-style methods (formerly NoteRegisterInator)
    #
    # Method names + signatures below are kept identical to the original
    # NoteRegisterInator so existing call sites don't need to change anything
    # except the import (NoteRegisterInator -> NoteEngramRepository).
    # -----------------------------------------------------------------------

    def register_inator(self, note_id: str, bubble_id: Optional[str], filepath: str):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO notes (id, bubble_id, filepath)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        last_analysed = CURRENT_TIMESTAMP
                    """,
                    (note_id, bubble_id, filepath),
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

    def fetch_uncached_notes_inator(self):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id AS note_id, bubble_id, filepath FROM notes WHERE cached = 0"
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def fetch_unanalysed_notes_inator(self):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id AS note_id, bubble_id, domain, subject, filepath
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

    def show_all_inator(self):
        conn = self._get_conn()
        try:
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
        content: str,
        bubble_id: Optional[str] = None,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        tags: Optional[list] = None,
    ):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO notes (id, bubble_id, domain, subject, topic, content, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        tags = excluded.tags,
                        topic = COALESCE(excluded.topic, notes.topic),
                        version = version + 1,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        note_id,
                        bubble_id,
                        domain,
                        subject,
                        topic,
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

    # -----------------------------------------------------------------------
    # Attempts
    # -----------------------------------------------------------------------

    def create_attempt(self, attempt: EngramAttempt) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engram_attempts
                  (id, engram_id, user_id, attempted_at, score, grader,
                   time_spent_ms, note_version, cognitive_level, context_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.id,
                    attempt.engram_id,
                    attempt.user_id,
                    attempt.attempted_at,
                    attempt.score,
                    attempt.grader.value,
                    attempt.time_spent_ms,
                    attempt.note_version,
                    attempt.cognitive_level,
                    (
                        json.dumps(attempt.context_snapshot)
                        if attempt.context_snapshot
                        else None
                    ),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def update_attempt_score(
        self, attempt_id: str, score: float, grader: GraderType
    ) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE engram_attempts SET score = ?, grader = ? WHERE id = ?",
                (score, grader.value, attempt_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_attempt_scores(
        self, engram_id: str, user_id: str, limit: int = 10
    ) -> list[float]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT score FROM engram_attempts
                WHERE engram_id = ? AND user_id = ? AND score IS NOT NULL
                ORDER BY attempted_at DESC LIMIT ?
                """,
                (engram_id, user_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return list(reversed([float(r["score"]) for r in rows]))

    # -----------------------------------------------------------------------
    # Type-specific responses
    # -----------------------------------------------------------------------

    def save_mcq_response(self, r: MCQResponse) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO mcq_responses
                  (attempt_id, selected_option, correct_option, is_correct, distractor_key)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    r.attempt_id,
                    r.selected_option,
                    r.correct_option,
                    int(r.is_correct),
                    r.distractor_key,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_flashcard_response(self, r: FlashcardResponse) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO flashcard_responses (attempt_id, self_rating, time_to_flip_ms)
                VALUES (?, ?, ?)
                """,
                (r.attempt_id, r.self_rating.value, r.time_to_flip_ms),
            )
            conn.commit()
        finally:
            conn.close()

    def save_short_answer_responses(self, responses: list[QuizResponse]) -> None:
        conn = self._get_conn()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO short_answer_responses
                  (id, attempt_id, question_index, selected_option, correct_option, is_correct)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.attempt_id,
                        r.question_index,
                        r.selected_option,
                        r.correct_option,
                        int(r.is_correct),
                    )
                    for r in responses
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def save_long_question_response(self, r: LongQuestionResponse) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO long_question_responses
                  (attempt_id, raw_answer, word_count, ai_feedback,
                   concepts_demonstrated, concepts_missed, misconceptions,
                   dimension_scores, level_demonstrated, regression_detected,
                   vector_id, graded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.attempt_id,
                    r.raw_answer,
                    r.word_count,
                    r.ai_feedback,
                    (
                        json.dumps(r.concepts_demonstrated)
                        if r.concepts_demonstrated
                        else None
                    ),
                    json.dumps(r.concepts_missed) if r.concepts_missed else None,
                    json.dumps(r.misconceptions) if r.misconceptions else None,
                    (
                        json.dumps(r.dimension_scores.to_dict())
                        if r.dimension_scores
                        else None
                    ),
                    r.level_demonstrated,
                    int(r.regression_detected),
                    r.vector_id,
                    r.graded_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_long_question_responses(
        self, engram_id: str, user_id: str, limit: int = 5
    ) -> list[LongQuestionResponse]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT lqr.* FROM long_question_responses lqr
                JOIN engram_attempts ea ON ea.id = lqr.attempt_id
                WHERE ea.engram_id = ? AND ea.user_id = ? AND lqr.graded_at IS NOT NULL
                ORDER BY ea.attempted_at DESC LIMIT ?
                """,
                (engram_id, user_id, limit),
            ).fetchall()
        finally:
            conn.close()

        result = []
        for row in rows:
            ds_raw = row["dimension_scores"]
            result.append(
                LongQuestionResponse(
                    attempt_id=row["attempt_id"],
                    raw_answer=row["raw_answer"],
                    word_count=row["word_count"] or 0,
                    ai_feedback=row["ai_feedback"],
                    concepts_demonstrated=(
                        json.loads(row["concepts_demonstrated"])
                        if row["concepts_demonstrated"]
                        else []
                    ),
                    concepts_missed=(
                        json.loads(row["concepts_missed"])
                        if row["concepts_missed"]
                        else []
                    ),
                    misconceptions=(
                        json.loads(row["misconceptions"])
                        if row["misconceptions"]
                        else []
                    ),
                    dimension_scores=(
                        DimensionScores.from_dict(json.loads(ds_raw))
                        if ds_raw
                        else None
                    ),
                    level_demonstrated=row["level_demonstrated"],
                    regression_detected=bool(row["regression_detected"]),
                    vector_id=row["vector_id"],
                    graded_at=row["graded_at"],
                )
            )
        return result

    # -----------------------------------------------------------------------
    # Mastery
    # -----------------------------------------------------------------------

    def get_mastery(self, engram_id: str, user_id: str) -> Optional[EngramMastery]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM engram_mastery WHERE engram_id = ? AND user_id = ?",
                (engram_id, user_id),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_mastery(row) if row else None

    def upsert_mastery(self, mastery: EngramMastery) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engram_mastery
                  (id, engram_id, user_id, state, current_score, stability,
                   interval_days, next_due_at, last_attempted_at, attempt_count,
                   lapse_count, consecutive_correct, current_level,
                   score_accuracy, score_depth, score_reasoning, score_connections,
                   score_originality, score_precision, score_awareness_of_limits,
                   updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(engram_id, user_id) DO UPDATE SET
                  state=excluded.state, current_score=excluded.current_score,
                  stability=excluded.stability, interval_days=excluded.interval_days,
                  next_due_at=excluded.next_due_at,
                  last_attempted_at=excluded.last_attempted_at,
                  attempt_count=excluded.attempt_count, lapse_count=excluded.lapse_count,
                  consecutive_correct=excluded.consecutive_correct,
                  current_level=excluded.current_level,
                  score_accuracy=excluded.score_accuracy, score_depth=excluded.score_depth,
                  score_reasoning=excluded.score_reasoning,
                  score_connections=excluded.score_connections,
                  score_originality=excluded.score_originality,
                  score_precision=excluded.score_precision,
                  score_awareness_of_limits=excluded.score_awareness_of_limits,
                  updated_at=excluded.updated_at
                """,
                (
                    mastery.id,
                    mastery.engram_id,
                    mastery.user_id,
                    mastery.state.value,
                    mastery.current_score,
                    mastery.stability,
                    mastery.interval_days,
                    mastery.next_due_at,
                    mastery.last_attempted_at,
                    mastery.attempt_count,
                    mastery.lapse_count,
                    mastery.consecutive_correct,
                    mastery.current_level,
                    mastery.score_accuracy,
                    mastery.score_depth,
                    mastery.score_reasoning,
                    mastery.score_connections,
                    mastery.score_originality,
                    mastery.score_precision,
                    mastery.score_awareness_of_limits,
                    mastery.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_topic_masteries(self, user_id: str, topic: str) -> list[EngramMastery]:
        """Matches MasteryRepository.get_topic_masteries; joins on notes.topic
        (what the note is about) — NOT notes.domain (pipeline classification)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT em.* FROM engram_mastery em
                JOIN engrams e ON e.id = em.engram_id
                JOIN notes n   ON n.id = e.note_id
                WHERE em.user_id = ? AND n.topic = ?
                """,
                (user_id, topic),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_mastery(r) for r in rows]

    def get_all_due_masteries(self, user_id: str) -> list[EngramMastery]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM engram_mastery
                WHERE user_id = ? AND state != 'suspended'
                  AND next_due_at <= datetime('now', '+1 day')
                ORDER BY next_due_at ASC
                """,
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_mastery(r) for r in rows]

    # -----------------------------------------------------------------------
    # Topic mastery (aggregate table; matches MasteryRepository's
    # upsert_topic_mastery / get_topic_mastery, and TopicMastery.topic)
    # -----------------------------------------------------------------------

    def upsert_topic_mastery(self, tm: TopicMastery) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO topic_mastery
                  (id, user_id, topic, factual_score, applied_score, conceptual_score,
                   doctoral_score, overall_score, engram_count, lapsed_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, topic) DO UPDATE SET
                  factual_score=excluded.factual_score,
                  applied_score=excluded.applied_score,
                  conceptual_score=excluded.conceptual_score,
                  doctoral_score=excluded.doctoral_score,
                  overall_score=excluded.overall_score,
                  engram_count=excluded.engram_count,
                  lapsed_count=excluded.lapsed_count,
                  updated_at=excluded.updated_at
                """,
                (
                    tm.id,
                    tm.user_id,
                    tm.topic,
                    tm.factual_score,
                    tm.applied_score,
                    tm.conceptual_score,
                    tm.doctoral_score,
                    tm.overall_score,
                    tm.engram_count,
                    tm.lapsed_count,
                    tm.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_topic_mastery(self, user_id: str, topic: str) -> Optional[TopicMastery]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM topic_mastery WHERE user_id = ? AND topic = ?",
                (user_id, topic),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return TopicMastery(
            id=row["id"],
            user_id=row["user_id"],
            topic=row["topic"],
            factual_score=row["factual_score"],
            applied_score=row["applied_score"],
            conceptual_score=row["conceptual_score"],
            doctoral_score=row["doctoral_score"],
            overall_score=row["overall_score"],
            engram_count=row["engram_count"],
            lapsed_count=row["lapsed_count"],
            updated_at=row["updated_at"],
        )

    # -----------------------------------------------------------------------
    # Misconceptions
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Grading jobs
    # -----------------------------------------------------------------------

    def create_grading_job(self, attempt_id: str, priority: int = 5) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO grading_jobs (id, attempt_id, status, priority) VALUES (?, ?, 'pending', ?)",
                (_id(), attempt_id, priority),
            )
            conn.commit()
        finally:
            conn.close()

    def update_grading_job(
        self, job_id: str, status: str, error: Optional[str] = None
    ) -> None:
        conn = self._get_conn()
        try:
            if status == "processing":
                conn.execute(
                    "UPDATE grading_jobs SET status=?, started_at=datetime('now'), attempts=attempts+1 WHERE id=?",
                    (status, job_id),
                )
            elif status == "done":
                conn.execute(
                    "UPDATE grading_jobs SET status=?, completed_at=datetime('now') WHERE id=?",
                    (status, job_id),
                )
            else:
                conn.execute(
                    "UPDATE grading_jobs SET status=?, error=? WHERE id=?",
                    (status, error, job_id),
                )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Engram generation queue
    # -----------------------------------------------------------------------

    def queue_engram_generation(
        self,
        note_id: str,
        user_id: str,
        trigger: str,
        target_congnitive_level: int,
        target_type: str,
        trigger_ref: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engram_generation_queue
                  (id, note_id, user_id, trigger, trigger_ref, target_congnitive_level, target_type, instructions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _id(),
                    note_id,
                    user_id,
                    trigger,
                    trigger_ref,
                    target_congnitive_level,
                    target_type,
                    instructions,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_pending_generation_jobs(self, limit: int = 10) -> list[dict]:
        """
        Returns up to `limit` pending rows from engram_generation_queue,
        oldest first. Each row is a plain dict with keys:
            id, note_id, user_id, trigger, trigger_ref,
            target_congnitive_level, target_type, instructions, status, created_at
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM engram_generation_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def mark_generation_job_done(self, job_id: str) -> None:
        """Mark a generation queue row as successfully completed."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE engram_generation_queue SET status = 'done' WHERE id = ?",
                    (job_id,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def mark_generation_job_failed(self, job_id: str, error: str) -> None:
        """Mark a generation queue row as failed, storing the error reason."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE engram_generation_queue
                    SET status = 'failed', instructions = COALESCE(instructions, '') || ? 
                    WHERE id = ?
                    """,
                    (f"\n[ERROR] {error}", job_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Engrams
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

    # -----------------------------------------------------------------------
    # Grading-worker support (fetch_pending_grading_jobs / get_grading_context)
    #
    # These back grading.worker.SQLiteWorkerLoop. grading_jobs CRUD already
    # existed (create_grading_job / update_grading_job); what was missing was
    # a way to (a) pull pending rows in priority order and (b) reassemble a
    # GradingJobPayload for one of them without the worker having to know
    # the engram_attempts -> engrams -> notes -> long_question_responses
    # join itself.
    # -----------------------------------------------------------------------

    def fetch_pending_grading_jobs(self, limit: int = 10) -> list[dict]:
        """Up to `limit` pending grading_jobs rows, highest priority first
        (ties broken oldest-first). Each row is a dict with keys: job_id,
        attempt_id, attempts, priority, created_at.

        `id` is aliased to `job_id` here because worker.run_worker_batch /
        WorkerLoop read job_row["job_id"], matching the same convention
        fetch_pending_generation_jobs uses for the generation queue.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id AS job_id, attempt_id, attempts, priority, created_at
                FROM grading_jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_grading_context(self, attempt_id: str) -> Optional[dict]:
        """Joins engram_attempts -> engrams -> notes ->
        long_question_responses for one attempt. Returns a dict with keys
        engram_id, user_id, cognitive_level, note_id, topic, raw_answer, or
        None if the attempt doesn't exist. The parsed LongQuestionContent
        itself is deliberately NOT reassembled here — get_engram() already
        knows how to join long_question_content + long_question_parts, so
        the caller (SQLiteWorkerLoop.hydrate_job) fetches that separately
        rather than duplicating the logic.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT ea.engram_id      AS engram_id,
                       ea.user_id        AS user_id,
                       ea.cognitive_level AS cognitive_level,
                       e.note_id         AS note_id,
                       n.topic           AS topic,
                       lqr.raw_answer    AS raw_answer
                FROM engram_attempts ea
                JOIN engrams e ON e.id = ea.engram_id
                LEFT JOIN notes n ON n.id = e.note_id
                LEFT JOIN long_question_responses lqr ON lqr.attempt_id = ea.id
                WHERE ea.id = ?
                """,
                (attempt_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_topic_engrams(self, user_id: str, topic: str) -> list[Engram]:
        """Matches MasteryRepository.get_topic_engrams(user_id, topic).

        Note: `user_id` is accepted for interface compatibility but currently
        unused — engrams have no user_id column (they're shared content;
        per-user state lives in engram_mastery). If per-user visibility rules
        for engrams are ever needed, join through engram_mastery here.

        Joins on notes.topic (what the note is about) — NOT notes.domain
        (the pipeline-tracking classification set by mark_analysed_inator)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT e.* FROM engrams e
                JOIN notes n ON n.id = e.note_id
                WHERE n.topic = ? AND e.is_active = 1
                """,
                (topic,),
            ).fetchall()
            return [self._row_to_engram(conn, r) for r in rows]
        finally:
            conn.close()

    def create_engram(self, engram: Engram) -> None:
        from dataclasses import asdict

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engrams (id, note_id, type, cognitive_level, tags, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    engram.id,
                    engram.note_id,
                    engram.type.value,
                    engram.cognitive_level,
                    json.dumps(engram.tags),
                    int(engram.is_active),
                ),
            )
            self._insert_typed_content(
                conn, engram.id, engram.type, asdict(engram.content)
            )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Type-specific engram creation — take the RAW dict(s) a generator
    # emits (matching engram_mcq_v1 / engram_flashcard_v1 / engram_short_answer_v1 /
    # engram_long_answer_v1 shapes) and write both the parent `engrams` row
    # and the type table in one call, so the generator pipeline doesn't
    # need to hand-build an Engram/*Content dataclass first. Each returns
    # the new engram_id.
    #
    # These are thin wrappers around create_engram's two-step insert
    # (engrams row + _insert_typed_content); create_engram itself is still
    # there for callers that already have a fully-built Engram dataclass.
    # -----------------------------------------------------------------------

    def _create_engram_row(
        self,
        conn: sqlite3.Connection,
        note_id: str,
        etype: "EngramType",
        cognitive_level: int,
        tags: Optional[list],
        engram_id: Optional[str],
    ) -> str:
        engram_id = engram_id or _id()
        # INSERT OR IGNORE: callers building a multi-question short_answer call
        # add_short_answer once per question, reusing the same engram_id so all
        # questions land under one engram. Only the first call should
        # actually create the engrams row; later calls with that same id
        # should just fall through to inserting their short_answer_questions row.
        conn.execute(
            """
            INSERT OR IGNORE INTO engrams (id, note_id, type, cognitive_level, tags, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (
                engram_id,
                note_id,
                etype.value,
                cognitive_level,
                json.dumps(tags or []),
            ),
        )
        return engram_id

    def add_mcq(
        self,
        note_id: str,
        data: dict,
        cognitive_level: int = 1,
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
                conn, note_id, EngramType.MCQ, cognitive_level, tags, engram_id
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
        data: dict,
        cognitive_level: int = 1,
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
                conn, note_id, EngramType.FLASHCARD, cognitive_level, tags, engram_id
            )
            self._insert_typed_content(conn, eid, EngramType.FLASHCARD, data)
            conn.commit()
            return eid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_short_answer(
        self,
        note_id: str,
        questions: list[dict],
        cognitive_level: int = 1,
        tags: Optional[list] = None,
        engram_id: Optional[str] = None,
    ) -> str:
        """Create a short_answer engram from the FULL raw engram_short_answer_v1 output
        array (QUIZ_SCHEMA's output — a list of question dicts, each with
        keys: finding_index, question_number, level, stem, expected_answer,
        hint, context_anchored, severity). One short_answer engram holds many
        short_answer_questions rows, one per question_index — mirrors the existing
        one-row-per-question_index shape."""
        conn = self._get_conn()
        try:
            eid = self._create_engram_row(
                conn, note_id, EngramType.QUIZ, cognitive_level, tags, engram_id
            )
            for q in questions:
                self._insert_typed_content(conn, eid, EngramType.QUIZ, q)
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
        data: dict,
        cognitive_level: int = 1,
        tags: Optional[list] = None,
        engram_id: Optional[str] = None,
    ) -> str:
        """Create a long_question (long_answer) engram from a raw
        engram_long_answer_v1 output dict (LFQ_SCHEMA's output — keys:
        finding_index, question_stem, answer, parts (list of
        part/level/question/marks/mark_scheme/note), severity,
        total_marks). Writes one long_question_content row plus one
        long_question_parts row per part."""
        conn = self._get_conn()
        try:
            eid = self._create_engram_row(
                conn,
                note_id,
                EngramType.LONG_QUESTION,
                cognitive_level,
                tags,
                engram_id,
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
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _row_to_mastery(row: sqlite3.Row) -> EngramMastery:
        return EngramMastery(
            id=row["id"],
            engram_id=row["engram_id"],
            user_id=row["user_id"],
            state=MasteryState(row["state"]),
            current_score=float(row["current_score"]),
            stability=float(row["stability"]),
            interval_days=float(row["interval_days"]),
            next_due_at=row["next_due_at"],
            last_attempted_at=row["last_attempted_at"],
            attempt_count=int(row["attempt_count"]),
            lapse_count=int(row["lapse_count"]),
            consecutive_correct=int(row["consecutive_correct"]),
            current_level=int(row["current_level"]),
            score_accuracy=float(row["score_accuracy"] or 0),
            score_depth=float(row["score_depth"] or 0),
            score_reasoning=float(row["score_reasoning"] or 0),
            score_connections=float(row["score_connections"] or 0),
            score_originality=float(row["score_originality"] or 0),
            score_precision=float(row["score_precision"] or 0),
            score_awareness_of_limits=float(row["score_awareness_of_limits"] or 0),
            updated_at=row["updated_at"],
        )

    def _row_to_engram(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Engram:
        etype = EngramType(row["type"])
        return Engram(
            id=row["id"],
            note_id=row["note_id"],
            type=etype,
            cognitive_level=int(row["cognitive_level"]),
            content=self._load_content(conn, row["id"], etype),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _load_content(
        self, conn: sqlite3.Connection, engram_id: str, etype: EngramType
    ):
        from ..core.types import (
            FlashcardContent,
            LongQuestionContent,
            LongQuestionPart,
            MCQContent,
            QuizContent,
            QuizQuestion,
        )

        if etype == EngramType.MCQ:
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

        if etype == EngramType.FLASHCARD:
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

        if etype == EngramType.QUIZ:
            rows = conn.execute(
                "SELECT * FROM short_answer_questions WHERE engram_id = ? ORDER BY question_index",
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

        # LONG_QUESTION
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
