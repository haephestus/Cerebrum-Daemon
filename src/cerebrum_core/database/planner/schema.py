"""
cerebrum_core.database.study_plan_registry.schema
=====================================================
All CREATE TABLE / CREATE INDEX statements for the study-plan DB, in one
place, executed via conn.executescript() from _base._ensure_schema().

The first three tables (study_plan_registry, plan_phase_registry,
plan_success_metric_registry) are unchanged from the original
StudyPlanRegisterInator._table_initiator_inator — copied verbatim so
existing rows/callers aren't affected by the mixin split.

plan_week_registry / plan_day_registry / plan_task_registry are new.
Tasks get their own child table (one row per task) rather than a
tasks_json blob on plan_day_registry, matching the existing convention
in note_registry's schema.py (short_question/long_question_parts are
child tables, not JSON arrays) — it's what lets complete_task_inator be
a plain `UPDATE ... WHERE task_id = ?` instead of JSON surgery.

topics on plan_week_registry stay as topics_json (a small array of
plain strings, joined against notes.topic / topic_mastery.topic in
note_registry.db). That join is cross-database — SQLite can't enforce
it as a real FOREIGN KEY, and no JOIN can span both files directly —
so it's resolved in Python at the service layer, not in SQL here.
"""

SCHEMA_SQL = """
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
);

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
);

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
);

-- ---------------------------------------------------------------------
-- Weekly/daily densification (new)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS plan_week_registry (
    week_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    phase_id INTEGER NOT NULL,
    week_number INTEGER NOT NULL,          -- absolute, 1-indexed from plan start
    focus_summary TEXT,
    topics_json TEXT NOT NULL DEFAULT '[]',  -- ["CRISPR Gene Drive in Mice", ...]
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | active | complete
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES study_plan_registry(plan_id)
        ON DELETE CASCADE,
    UNIQUE (plan_id, week_number)
);

CREATE TABLE IF NOT EXISTS plan_day_registry (
    day_id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,          -- 0=Mon .. 6=Sun, matches weekly_rhythm_json
    FOREIGN KEY (week_id) REFERENCES plan_week_registry(week_id)
        ON DELETE CASCADE,
    UNIQUE (week_id, day_of_week)
);

CREATE TABLE IF NOT EXISTS plan_task_registry (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    day_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    task_type TEXT NOT NULL,               -- study | practice | build | review | milestone_check
    topic TEXT,                            -- NULL for build/milestone_check
    target_minutes INTEGER,
    source_hint TEXT,                      -- why this task exists (mastery gap, misconception, etc.)
    status TEXT NOT NULL DEFAULT 'pending', -- pending | complete
    auto_resolved INTEGER NOT NULL DEFAULT 0,  -- 1 if completion came from engram_attempts, not a manual tap
    completed_at TIMESTAMP,
    FOREIGN KEY (day_id) REFERENCES plan_day_registry(day_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_plan_week_phase ON plan_week_registry(plan_id, phase_id);
CREATE INDEX IF NOT EXISTS idx_plan_day_week ON plan_day_registry(week_id);
CREATE INDEX IF NOT EXISTS idx_plan_task_day ON plan_task_registry(day_id);
CREATE INDEX IF NOT EXISTS idx_plan_task_topic ON plan_task_registry(topic);
"""
