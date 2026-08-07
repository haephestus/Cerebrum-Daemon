#!/usr/bin/env python3
"""
migrate_add_note_owner.py
===========================
One-time migration for an existing note_registry.db that predates the
`notes.user_id` column. Backfills every existing note to a single owner
(appropriate when there's been exactly one user so far).

Steps, all inside one transaction so it's atomic (either fully applies or
leaves the DB untouched on error):

  1. Confirm the target user_id exists in `users` (refuses to guess/create
     one — you should already have a users row for whoever owns these notes).
  2. Add `user_id` as a nullable column (SQLite can't add NOT NULL to an
     ALTER TABLE ADD COLUMN when the table already has rows, unless every
     row gets the same constant default — and we want the FK-checked
     value, not a bare default).
  3. Backfill every existing row to the given user_id.
  4. Rebuild the table (the standard SQLite "12-step" ALTER pattern) so
     the final column is NOT NULL REFERENCES users(id), matching
     schema.py exactly — so future `_ensure_schema()` calls (which use
     CREATE TABLE IF NOT EXISTS) see an up-to-date table and no-op.
  5. Recreate every index on notes.

Idempotent: if notes.user_id already exists, the script reports that and
exits without touching anything.

Usage:
    python migrate_add_note_owner.py --db /path/to/note_registry.db --user-id u1
    python migrate_add_note_owner.py --db /path/to/note_registry.db --user-id u1 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_NOTES_COLUMNS_OLD = [
    "id", "bubble_id", "domain", "subject", "topic", "cached", "analysed",
    "analysis_status", "analysis_error", "filepath", "content", "tags",
    "version", "created_at", "updated_at", "last_analysed",
]

_REBUILD_SQL = """
CREATE TABLE notes_new (
  id            TEXT PRIMARY KEY,
  user_id       TEXT NOT NULL REFERENCES users(id),
  bubble_id     TEXT,
  domain        TEXT,
  subject       TEXT,
  topic         TEXT,
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

INSERT INTO notes_new
  (id, user_id, bubble_id, domain, subject, topic, cached, analysed,
   analysis_status, analysis_error, filepath, content, tags, version,
   created_at, updated_at, last_analysed)
SELECT
   id, user_id, bubble_id, domain, subject, topic, cached, analysed,
   analysis_status, analysis_error, filepath, content, tags, version,
   created_at, updated_at, last_analysed
FROM notes;

DROP TABLE notes;
ALTER TABLE notes_new RENAME TO notes;

CREATE INDEX IF NOT EXISTS idx_notes_user_id  ON notes(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_domain   ON notes(domain);
CREATE INDEX IF NOT EXISTS idx_notes_topic    ON notes(topic);
CREATE INDEX IF NOT EXISTS idx_notes_tags     ON notes(tags);
CREATE INDEX IF NOT EXISTS idx_notes_cached   ON notes(cached);
CREATE INDEX IF NOT EXISTS idx_notes_analysed ON notes(analysed);
"""


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _backup(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = db_path.with_suffix(db_path.suffix + f".pre-user-id-migration.{stamp}.bak")
    shutil.copy2(db_path, backup_path)
    # WAL/SHM files, if present, aren't strictly needed for the backup to be
    # restorable (sqlite checkpoints on close), but copy them too if they
    # exist so a restore doesn't require a checkpoint step.
    for ext in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + ext)
        if side.exists():
            shutil.copy2(side, backup_path.with_name(backup_path.name + ext))
    return backup_path


def migrate(db_path: Path, user_id: str, dry_run: bool = False) -> None:
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")  # off during the rebuild itself

    try:
        existing_cols = _column_names(conn, "notes")
        if "user_id" in existing_cols:
            print("notes.user_id already exists — nothing to do. Exiting.")
            return

        missing = set(_NOTES_COLUMNS_OLD) - set(existing_cols)
        if missing:
            print(
                f"WARNING: notes table doesn't look like the expected pre-migration "
                f"shape (missing columns: {sorted(missing)}). Proceeding anyway, but "
                f"double check this is the right database.",
                file=sys.stderr,
            )

        user_row = conn.execute(
            "SELECT id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user_row:
            print(
                f"ERROR: no row in `users` with id = {user_id!r}. Create that user "
                f"first (repo.create_user(...)) — this script won't fabricate one, "
                f"since it doesn't know the right name/email.",
                file=sys.stderr,
            )
            sys.exit(1)

        note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        print(f"Found {note_count} existing note(s) to assign to user_id={user_id!r}.")

        if dry_run:
            print("Dry run — no changes made. Re-run without --dry-run to apply.")
            return

        backup_path = _backup(db_path)
        print(f"Backed up database to {backup_path}")

        conn.execute("BEGIN")
        conn.execute("ALTER TABLE notes ADD COLUMN user_id TEXT REFERENCES users(id)")
        conn.execute("UPDATE notes SET user_id = ?", (user_id,))

        still_null = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE user_id IS NULL"
        ).fetchone()[0]
        if still_null:
            raise RuntimeError(
                f"{still_null} row(s) still have NULL user_id after backfill — aborting."
            )

        conn.executescript(_REBUILD_SQL)

        fk_problems = conn.execute("PRAGMA foreign_key_check(notes)").fetchall()
        if fk_problems:
            raise RuntimeError(f"Foreign key check failed after rebuild: {fk_problems}")

        conn.commit()

        final_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        final_cols = _column_names(conn, "notes")
        print(f"Migration complete. notes row count: {note_count} -> {final_count}")
        print(f"notes columns now: {final_cols}")

    except Exception:
        conn.rollback()
        print("Migration FAILED and was rolled back. Database unchanged.", file=sys.stderr)
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Path to note_registry.db")
    parser.add_argument("--user-id", required=True, help="user_id to assign all existing notes to")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen, make no changes")
    args = parser.parse_args()

    migrate(args.db, args.user_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
