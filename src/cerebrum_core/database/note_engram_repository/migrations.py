"""
cerebrum_core.database.note_engram_repository.migrations
=========================================================
Lightweight, in-house migration runner for note_registry.db.

Schema is still created by SCHEMA_SQL (CREATE TABLE IF NOT EXISTS) — that
handles fresh databases. This handles *existing* databases whose shape
predates a schema change, formalising the previously ad-hoc pattern (a
one-off migrate_add_note_owner script, plus inline ALTERs in the chunk
registries) into one ordered, tracked list run automatically at schema
bootstrap.

Rules:
  * Each migration is idempotent and guarded, so it's a no-op on a fresh DB
    (where SCHEMA_SQL already produced the corrected shape) and applies once
    on an existing DB.
  * Applied ids are recorded in schema_migrations; a migration never runs
    twice.
  * Append new migrations to the end. Never reorder, rename, or mutate an
    existing id — that's what keeps already-migrated DBs consistent.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable

logger = logging.getLogger(__name__)

Migration = tuple[str, str, Callable[[sqlite3.Connection], None]]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _rename_generation_queue_cognitive_typo(conn: sqlite3.Connection) -> None:
    """Rename engram_generation_queue.target_congnitive_level (typo) to
    target_cognitive_level. Guarded: only acts when the old column exists
    and the new one doesn't, so a fresh DB (already correct) is untouched."""
    cols = _columns(conn, "engram_generation_queue")
    if not cols:
        return  # table absent (shouldn't happen after SCHEMA_SQL) — nothing to do
    if "target_congnitive_level" in cols and "target_cognitive_level" not in cols:
        conn.execute(
            "ALTER TABLE engram_generation_queue "
            "RENAME COLUMN target_congnitive_level TO target_cognitive_level"
        )
        logger.info(
            "migration: renamed engram_generation_queue.target_congnitive_level "
            "-> target_cognitive_level"
        )


def _introduce_topic_entity(conn: sqlite3.Connection) -> None:
    """Backfill the topics entity on an existing DB: add notes.topic_id /
    topic_mastery.topic_id (SCHEMA_SQL only adds them to fresh tables), then
    create one topic row per distinct (user, canonical-slug) seen in the
    denormalised topic strings and link every row to it. Collapses any
    pre-existing spelling variants of the same topic into one entity.

    The topics table itself is created by SCHEMA_SQL (CREATE IF NOT EXISTS),
    which runs before migrations on every bootstrap, so it exists here.
    """
    import uuid

    from cerebrum_core.utils.topic_inator import normalize_topic, topic_slug

    if "topic_id" not in _columns(conn, "notes"):
        conn.execute("ALTER TABLE notes ADD COLUMN topic_id TEXT")
    if "topic_id" not in _columns(conn, "topic_mastery"):
        conn.execute("ALTER TABLE topic_mastery ADD COLUMN topic_id TEXT")

    # (user_id, slug) -> (topic_id, canonical_name), built as we go so the
    # same topic seen in both notes and topic_mastery maps to one entity.
    resolved: dict[tuple[str, str], tuple[str, str]] = {}

    def ensure_topic(user_id: str, raw_name: str) -> "str | None":
        slug = topic_slug(raw_name)
        if not user_id or not slug:
            return None
        key = (user_id, slug)
        if key in resolved:
            return resolved[key][0]
        row = conn.execute(
            "SELECT id, name FROM topics WHERE user_id = ? AND slug = ?",
            (user_id, slug),
        ).fetchone()
        if row:
            resolved[key] = (row[0], row[1])
            return row[0]
        topic_id = uuid.uuid4().hex
        canonical = normalize_topic(raw_name)
        conn.execute(
            "INSERT INTO topics (id, user_id, slug, name) VALUES (?, ?, ?, ?)",
            (topic_id, user_id, slug, canonical),
        )
        resolved[key] = (topic_id, canonical)
        return topic_id

    def backfill(table: str) -> None:
        rows = conn.execute(
            f"SELECT id, user_id, topic FROM {table} "
            "WHERE topic IS NOT NULL AND topic <> '' AND topic_id IS NULL"
        ).fetchall()
        for row_id, user_id, raw in rows:
            topic_id = ensure_topic(user_id, raw)
            if topic_id is None:
                continue
            canonical = resolved[(user_id, topic_slug(raw))][1]
            conn.execute(
                f"UPDATE {table} SET topic_id = ?, topic = ? WHERE id = ?",
                (topic_id, canonical, row_id),
            )

    backfill("notes")
    backfill("topic_mastery")

    # Indexes on topic_id live here rather than in SCHEMA_SQL: on an existing
    # DB the column only exists after the ALTER above, and SCHEMA_SQL runs
    # before migrations. IF NOT EXISTS keeps this a no-op on fresh DBs.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_topic_id ON notes(topic_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_topic_mastery_topic_id "
        "ON topic_mastery(topic_id)"
    )

    logger.info(
        "migration: introduced topic entity — %d topic(s) created/linked",
        len(resolved),
    )


def _add_user_password_hash(conn: sqlite3.Connection) -> None:
    """Add users.password_hash on an existing DB (SCHEMA_SQL only adds it to
    fresh tables). Guarded: no-op once the column is present. Pre-existing
    rows get '' (the column default), which no login can match — those legacy
    accounts must set a password before they can authenticate."""
    cols = _columns(conn, "users")
    if not cols:
        return  # table absent (shouldn't happen after SCHEMA_SQL) — nothing to do
    if "password_hash" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''"
        )
        logger.info("migration: added users.password_hash")


# Ordered migration list. Ids are the source of truth in schema_migrations.
_MIGRATIONS: list[Migration] = [
    (
        "0001_rename_generation_queue_cognitive_typo",
        "Fix engram_generation_queue cognitive-level column name typo",
        _rename_generation_queue_cognitive_typo,
    ),
    (
        "0002_introduce_topic_entity",
        "Add topics table links (notes.topic_id / topic_mastery.topic_id) and backfill",
        _introduce_topic_entity,
    ),
    (
        "0003_add_user_password_hash",
        "Add users.password_hash for credential-backed accounts",
        _add_user_password_hash,
    ),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every not-yet-applied migration in order, recording each in
    schema_migrations. Called from _RepositoryBase._ensure_schema after
    SCHEMA_SQL has run, on the same connection/lock."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id         TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    applied = {r[0] for r in conn.execute("SELECT id FROM schema_migrations").fetchall()}

    for mid, desc, fn in _MIGRATIONS:
        if mid in applied:
            continue
        try:
            fn(conn)
            conn.execute("INSERT INTO schema_migrations (id) VALUES (?)", (mid,))
            conn.commit()
            logger.info("migration applied: %s (%s)", mid, desc)
        except Exception:
            conn.rollback()
            logger.exception("migration failed: %s — %s", mid, desc)
            raise
