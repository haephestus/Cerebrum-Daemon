"""
cerebrum_core.engrams.storage.note_engram_repository.schema
==============================================================
Single source of truth for both note tracking and engrams/mastery.
All CREATE TABLE statements use IF NOT EXISTS, so re-running against an
existing DB is safe.

Schema change vs the old `notes` table: `title`, `subtopic`, `source`, and
`embedding_id` are dropped. `topic` is kept — it's what the note is about,
and is genuinely distinct from `domain`/`subject` (the pipeline-tracking
classification that note_registry's mark_analysed_inator sets). Engrams and
mastery group on `topic`, never on `domain`.

CONTENT-TABLE SHAPE (2024 revision): mcq_content, flashcard_content,
short_question, and long_question_content/long_question_parts below are
shaped to match the ACTUAL generator output schemas (engram_mcq_v1,
engram_flashcard_v1, engram_short_question_v1, engram_long_question_v1)
rather than an earlier, incorrect assumption that all four content types
were MCQ-like with a rubric. See inline comments on each table for what
changed and why.
"""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------------
-- topics — the topic ENTITY. `topic` used to be a free-text string that
-- engrams/mastery grouped on and the study planner cross-DB-joined on,
-- so two spellings of the same subject fragmented a student's mastery.
-- Each (user, canonical-slug) now gets one stable id here; slug is the
-- casefolded/normalised identity key (see topic_inator.topic_slug), name
-- keeps its first-seen display casing. notes.topic_id / topic_mastery.
-- topic_id reference this; the denormalised name columns are kept for
-- display and so existing string-keyed reads keep working.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topics (
  id         TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL REFERENCES users(id),
  slug       TEXT NOT NULL,
  name       TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(user_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id);

CREATE TABLE IF NOT EXISTS notes (
  id            TEXT PRIMARY KEY,     -- was note_registry.note_id
  user_id       TEXT NOT NULL REFERENCES users(id),  -- owner; every note belongs to exactly one user
  bubble_id     TEXT,
  domain        TEXT,   -- pipeline tracking classification, set by mark_analysed_inator
  subject       TEXT,   -- pipeline tracking classification, set by mark_analysed_inator
  topic         TEXT,   -- denormalised canonical display name (= topics.name)
  topic_id      TEXT REFERENCES topics(id),  -- authoritative topic entity link
  cached        INTEGER NOT NULL DEFAULT 0,
  analysed      INTEGER NOT NULL DEFAULT 0,
  analysis_status TEXT NOT NULL DEFAULT 'not_started'
                 CHECK(analysis_status IN ('not_started','pending','running','done','failed')),
  analysis_error  TEXT,
  filepath      TEXT,
  content       TEXT NOT NULL DEFAULT '',
  tags          TEXT NOT NULL DEFAULT '[]',
  version       INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
  last_analysed TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_domain  ON notes(domain);
CREATE INDEX IF NOT EXISTS idx_notes_topic   ON notes(topic);
-- idx_notes_topic_id is created by migration 0002, not here: on an existing
-- DB the notes table pre-dates topic_id, and CREATE TABLE IF NOT EXISTS
-- won't add the column, so indexing it in SCHEMA_SQL (which runs before
-- migrations) would fail with "no such column". The migration adds the
-- column then the index; on a fresh DB the migration just creates the index
-- (IF NOT EXISTS) since the column is already present from CREATE TABLE.
CREATE INDEX IF NOT EXISTS idx_notes_tags    ON notes(tags);
CREATE INDEX IF NOT EXISTS idx_notes_cached  ON notes(cached);
CREATE INDEX IF NOT EXISTS idx_notes_analysed ON notes(analysed);

CREATE TABLE IF NOT EXISTS engrams (
  id              TEXT PRIMARY KEY,
  bubble_id       TEXT NOT NULL,
  note_id         TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  type            TEXT NOT NULL CHECK(type IN ('mcq','flashcard','short_question','long_question')),
  target_cognitive_level INTEGER NOT NULL DEFAULT 1 CHECK(target_cognitive_level BETWEEN 1 AND 7),
  tags            TEXT NOT NULL DEFAULT '[]',
  is_active       INTEGER NOT NULL DEFAULT 1,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_engrams_note_id         ON engrams(note_id);
CREATE INDEX IF NOT EXISTS idx_engrams_type            ON engrams(type);
CREATE INDEX IF NOT EXISTS idx_engrams_target_cognitive_level ON engrams(target_cognitive_level);

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
-- short_question — matches engram_short_question_v1 output.
--
-- Changed vs previous revision: this was MCQ-shaped (option_a-d,
-- correct_option), but the generator produces OPEN-RESPONSE
-- recall/explain/apply questions with a free-text expected_answer to be
-- graded (by an LLM grader), not a lettered choice. option_a-d and
-- correct_option are DROPPED; level, stem, expected_answer, hint,
-- context_anchored, severity, finding_index ADDED.
--
-- RESOLVED (async grading pass): short_question_responses below now
-- stores a free-text raw_answer per sub-question plus AI-graded fields,
-- mirroring long_question_responses. Open-response short answers are
-- graded asynchronously by the grading worker (see mastery_service.
-- submit_short_question / grading.worker), exactly like long_question.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS short_question (
  id               TEXT PRIMARY KEY,
  engram_id        TEXT NOT NULL REFERENCES engrams(id) ON DELETE CASCADE,
  finding_index    INTEGER,
  question_index   INTEGER NOT NULL,
  level            TEXT NOT NULL CHECK(level IN ('recall','understand','apply','analyse', 'synthesise','evaluate','doctoral')),
  stem             TEXT NOT NULL,
  expected_answer  TEXT NOT NULL,
  hint             TEXT,
  context_anchored INTEGER NOT NULL DEFAULT 0,
  severity         TEXT,
  UNIQUE(engram_id, question_index)
);

CREATE INDEX IF NOT EXISTS idx_short_question_engram ON short_question(engram_id, question_index);

-- ---------------------------------------------------------------------
-- long_question_content / long_question_parts — matches
-- engram_long_question_v1 output.
--
-- Changed vs previous revision: the old table assumed one flat question
-- with one holistic multi-dimension rubric (rubric_accuracy, ...,
-- rubric_awareness_of_limits). The generator instead produces ONE
-- scaffolded multi-part question (part a/b/c/..., each independently
-- marked with its own `marks` + `mark_scheme`) — a one-to-many shape,
-- not one-to-one, and there is no holistic rubric at all in the actual
-- output. rubric_* columns DROPPED from the parent; a new child table
-- long_question_parts holds one row per part, mirroring how
-- short_question already does one row per question_index.
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
  level       TEXT NOT NULL CHECK(level IN ('recall','understand','apply','analyse','synthesise','evaluate','doctoral')),
  question    TEXT NOT NULL,
  marks       INTEGER NOT NULL,
  mark_scheme TEXT NOT NULL,
  note        TEXT,
  UNIQUE(engram_id, part)
);

CREATE INDEX IF NOT EXISTS idx_long_question_parts_engram ON long_question_parts(engram_id);

CREATE TABLE IF NOT EXISTS users (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  email         TEXT UNIQUE NOT NULL,
  -- bcrypt hash of the account password. NOT NULL so identity is always
  -- credential-backed; DEFAULT '' exists only so migration 0003 can add the
  -- column to a pre-existing DB (those legacy rows must reset their password).
  password_hash TEXT NOT NULL DEFAULT '',
  settings      TEXT NOT NULL DEFAULT '{}',
  created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Organisations: a named group users can belong to, used to scope shared
-- resources (e.g. knowledgebase files) to a set of people rather than one
-- owner. New tables, so CREATE IF NOT EXISTS covers both fresh and existing
-- DBs — no migration needed.
CREATE TABLE IF NOT EXISTS orgs (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Membership join table. ON DELETE CASCADE on user_id means a user's rows
-- here vanish when the user is deleted (so delete_user needs no explicit
-- cleanup); CASCADE on org_id means deleting an org drops its memberships.
CREATE TABLE IF NOT EXISTS org_members (
  org_id    TEXT NOT NULL REFERENCES orgs(id)  ON DELETE CASCADE,
  user_id   TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role      TEXT NOT NULL DEFAULT 'member' CHECK(role IN ('owner','admin','member')),
  joined_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (org_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id);

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
  target_cognitive_level  INTEGER NOT NULL,
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

-- short_question_responses — one row per sub-question answered within an
-- attempt (short_question engrams hold many question_index rows; see the
-- short_question content table above). Reshaped for open-response, async
-- AI grading to mirror long_question_responses:
--   * selected_option/correct_option DROPPED — short questions are
--     free-text recall/understand/apply answers, not lettered choices.
--   * raw_answer holds the student's text (NOT NULL: they always submit
--     something for each question they answer).
--   * score/is_correct/feedback/misconceptions/graded_at are NULL until
--     the grading worker lands the AI grade for this attempt. is_correct
--     is a convenience flag (score >= a pass threshold), not an
--     exact-match — grading is per-question float scoring.
--   * UNIQUE(attempt_id, question_index) so a re-submit of the same
--     question within an attempt replaces rather than duplicates.
CREATE TABLE IF NOT EXISTS short_question_responses (
  id              TEXT PRIMARY KEY,
  attempt_id      TEXT NOT NULL REFERENCES engram_attempts(id),
  question_index  INTEGER NOT NULL,
  raw_answer      TEXT NOT NULL,
  score           REAL,
  is_correct      INTEGER,
  feedback        TEXT,
  misconceptions  TEXT,
  graded_at       TEXT,
  UNIQUE(attempt_id, question_index)
);

CREATE INDEX IF NOT EXISTS idx_sqr_attempt ON short_question_responses(attempt_id);

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
  topic            TEXT NOT NULL,   -- denormalised canonical display name (= topics.name)
  topic_id         TEXT REFERENCES topics(id),  -- authoritative topic entity link
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
-- idx_topic_mastery_topic_id is created by migration 0002 (see idx_notes_topic_id note above).

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
  bubble_id    TEXT NOT NULL,
  note_id      TEXT NOT NULL REFERENCES notes(id),
  user_id      TEXT NOT NULL REFERENCES users(id),
  trigger      TEXT NOT NULL,
  trigger_ref  TEXT,
  target_cognitive_level INTEGER NOT NULL,
  target_type  TEXT NOT NULL,
  instructions TEXT,
  status       TEXT NOT NULL DEFAULT 'pending',
  attempts     INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Learning profile, declared layer (the Bayesian *prior*): what the user says
-- about how they want to be taught. One row per user; `axes` is a JSON map of
-- the fixed dimension names (see cerebrum_core.learning_profile_inator.AXES) to
-- a scalar in [-1, 1]. Deliberately separate from users.settings so the inferred
-- layer can sit beside it and be reasoned about independently -- declared is
-- never overwritten by inference. New table, so CREATE IF NOT EXISTS covers
-- fresh + existing DBs; no migration needed.
CREATE TABLE IF NOT EXISTS learning_profile_declared (
  user_id    TEXT PRIMARY KEY REFERENCES users(id),
  axes       TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Learning profile, inferred layer (the *evidence*): an append-only log of
-- behavioural signals, each pulling one axis in a direction with a weight =
-- how strong/confident that signal is. The inferred posterior (mean +
-- confidence per axis) is DERIVED from these rows on demand by
-- learning_profile_inator -- there is intentionally no cached posterior table,
-- so this log is the single source of truth AND the growth-trajectory record.
-- `source` = origin ('note_analysis' | 'engram_performance'); `ref` = an audit
-- pointer (e.g. note_id / attempt id).
CREATE TABLE IF NOT EXISTS learning_profile_evidence (
  id         TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL REFERENCES users(id),
  source     TEXT NOT NULL,
  axis       TEXT NOT NULL,
  value      REAL NOT NULL,
  weight     REAL NOT NULL DEFAULT 1.0,
  ref        TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_lp_evidence_user_axis
  ON learning_profile_evidence(user_id, axis);

-- Suggested readings (gap 3). A persisted candidate reading for a user, keyed
-- by the seed that produced it (a note_id or topic). `source` is 'knowledgebase'
-- for Tier-1 (already in the KB) or a provider name for external tiers.
-- `status` tracks the candidate → accepted → ingested lifecycle; `in_kb` +
-- `file_fingerprint` are set once a reading lives in the knowledge base.
-- `addresses` is a JSON list of the weak-areas/gaps the reading speaks to.
-- New table → CREATE IF NOT EXISTS covers fresh + existing DBs; no migration.
CREATE TABLE IF NOT EXISTS suggested_readings (
  id               TEXT PRIMARY KEY,
  user_id          TEXT NOT NULL REFERENCES users(id),
  seed_ref         TEXT NOT NULL,
  title            TEXT NOT NULL,
  source           TEXT NOT NULL,
  url              TEXT,
  file_fingerprint TEXT,
  license          TEXT,
  snippet          TEXT,
  reason           TEXT,
  addresses        TEXT NOT NULL DEFAULT '[]',
  score            REAL NOT NULL DEFAULT 0.0,
  in_kb            INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL DEFAULT 'candidate',
  created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_suggested_readings_user_seed
  ON suggested_readings(user_id, seed_ref);

"""
