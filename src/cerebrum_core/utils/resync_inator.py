#!/usr/bin/env python3
"""
resync_notes_from_disk.py
============================
Recovery script for when notes.id rows are gone but the note files on
disk (and everything downstream that references notes.id — engrams,
engram_mastery, engram_attempts, etc.) are still intact.

Walks:
    <base_dir>/<bubble_id>/notes/<note_id>/content.json

and re-inserts one `notes` row per note_id found, using the folder name
as the note's id. Because engrams.note_id (and everything that joins
through it) was never deleted, re-inserting a notes row with the SAME id
as before automatically reconnects it to its existing engrams/mastery/
attempts — nothing needs to be regenerated.

Standalone (raw sqlite3 + os.walk, no cerebrum_core import) on purpose:
during a recovery you don't want an unrelated import error somewhere
else in the app blocking you from fixing the actual problem.

Skips:
  - any bubble/note directory starting with "." (e.g. notes/.archives)
  - a bubble directory with no notes/ subdirectory
  - a note directory with no content.json (reported, not silently dropped)

content.json's actual note-text field name isn't assumed — the script
tries a few common keys (content, text, body, markdown) and falls back
to storing the raw JSON as a string if none match, so nothing is lost
even if the guess is wrong. It reports which key it used per note in
--verbose mode so you can sanity-check the first run.

Usage:
    python resync_notes_from_disk.py --db /path/to/note_registry.db --user-id u1 --dry-run
    python resync_notes_from_disk.py --db /path/to/note_registry.db --user-id u1
    python resync_notes_from_disk.py --db /path/to/note_registry.db --user-id u1 \\
        --base-dir ~/.local/share/cerebrum/study_bubbles
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_CONTENT_KEYS = ("content", "text", "body", "markdown")
_DEFAULT_BASE_DIR = Path.home() / ".local/share/cerebrum/study_bubbles"


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _backup(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = db_path.with_suffix(db_path.suffix + f".pre-notes-resync.{stamp}.bak")
    shutil.copy2(db_path, backup_path)
    for ext in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + ext)
        if side.exists():
            shutil.copy2(side, backup_path.with_name(backup_path.name + ext))
    return backup_path


def _extract_content(content_json_path: Path, verbose: bool) -> tuple[str, str]:
    """Returns (content_text, key_used_for_reporting)."""
    try:
        raw = content_json_path.read_text()
        data = json.loads(raw)
    except Exception as e:
        return "", f"<unreadable: {e}>"

    if isinstance(data, dict):
        for key in _CONTENT_KEYS:
            if key in data and isinstance(data[key], str):
                return data[key], key

    # No recognizable text field — fall back to storing the raw JSON so
    # nothing is lost, and flag it clearly for a follow-up look.
    return raw, "<raw json fallback>"


def _extract_bubble_topic(bubble_dir: Path) -> str | None:
    info_path = bubble_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text())
    except Exception:
        return None
    if isinstance(data, dict):
        for key in ("name", "title", "topic"):
            if key in data and isinstance(data[key], str):
                return data[key]
    return None


def find_notes_on_disk(base_dir: Path, verbose: bool) -> list[dict]:
    found = []
    if not base_dir.exists():
        print(f"ERROR: base dir {base_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    for bubble_dir in sorted(base_dir.iterdir()):
        if not bubble_dir.is_dir() or bubble_dir.name.startswith("."):
            continue
        bubble_id = bubble_dir.name
        notes_dir = bubble_dir / "notes"
        if not notes_dir.exists():
            if verbose:
                print(f"  [skip] {bubble_id}: no notes/ subdirectory")
            continue

        topic = _extract_bubble_topic(bubble_dir)

        for note_dir in sorted(notes_dir.iterdir()):
            if not note_dir.is_dir() or note_dir.name.startswith("."):
                continue
            note_id = note_dir.name
            content_json = note_dir / "content.json"
            if not content_json.exists():
                print(
                    f"  [WARN] {bubble_id}/{note_id}: no content.json — "
                    f"registering with empty content",
                    file=sys.stderr,
                )
                content, key_used = "", "<missing content.json>"
            else:
                content, key_used = _extract_content(content_json, verbose)

            if verbose:
                print(f"  [found] {bubble_id}/{note_id} (content key: {key_used})")

            found.append(
                {
                    "note_id": note_id,
                    "bubble_id": bubble_id,
                    "topic": topic,
                    "filepath": str(content_json),
                    "content": content,
                }
            )
    return found


def resync(
    db_path: Path, user_id: str, base_dir: Path, dry_run: bool, verbose: bool
) -> None:
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        if not _table_exists(conn, "notes"):
            print(
                "ERROR: notes table doesn't exist at all — this script only "
                "re-inserts rows into an existing table. Run your schema "
                "setup (_ensure_schema / CREATE TABLE) first.",
                file=sys.stderr,
            )
            sys.exit(1)

        cols = _column_names(conn, "notes")
        if "user_id" not in cols:
            print(
                "ERROR: notes table has no user_id column — run the "
                "user_id migration before this script.",
                file=sys.stderr,
            )
            sys.exit(1)

        user_row = conn.execute(
            "SELECT id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user_row:
            print(
                f"ERROR: no row in `users` with id = {user_id!r}. Create that "
                f"user first — this script won't fabricate one.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Scanning {base_dir} ...")
        notes = find_notes_on_disk(base_dir, verbose)
        print(
            f"Found {len(notes)} note(s) on disk across "
            f"{len({n['bubble_id'] for n in notes})} bubble(s)."
        )

        existing_ids = {r[0] for r in conn.execute("SELECT id FROM notes").fetchall()}
        already_present = [n for n in notes if n["note_id"] in existing_ids]
        to_insert = [n for n in notes if n["note_id"] not in existing_ids]
        print(f"  {len(to_insert)} missing from DB (will be (re)inserted)")
        print(f"  {len(already_present)} already present (will be left alone)")

        if dry_run:
            print("\nDry run — no changes made. Sample of what would be inserted:")
            for n in to_insert[:10]:
                print(
                    f"    id={n['note_id']} bubble_id={n['bubble_id']} "
                    f"topic={n['topic']!r} filepath={n['filepath']}"
                )
            if len(to_insert) > 10:
                print(f"    ... and {len(to_insert) - 10} more")
            return

        if not to_insert:
            print("Nothing to insert. Exiting.")
            return

        backup_path = _backup(db_path)
        print(f"Backed up database to {backup_path}")

        conn.execute("BEGIN")
        for n in to_insert:
            conn.execute(
                """
                INSERT INTO notes (id, user_id, bubble_id, topic, filepath, content)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    bubble_id = excluded.bubble_id,
                    topic     = COALESCE(excluded.topic, notes.topic),
                    filepath  = excluded.filepath,
                    content   = excluded.content,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    n["note_id"],
                    user_id,
                    n["bubble_id"],
                    n["topic"],
                    n["filepath"],
                    n["content"],
                ),
            )
        conn.commit()

        final_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        print(
            f"Inserted {len(to_insert)} note(s). notes table now has {final_count} row(s)."
        )

        # Report how many engrams got reconnected as a sanity check, if the
        # engrams table exists.
        if _table_exists(conn, "engrams"):
            reconnected = conn.execute(
                """
                SELECT COUNT(*) FROM engrams e
                JOIN notes n ON n.id = e.note_id
                WHERE e.note_id IN ({})
                """.format(
                    ",".join("?" * len(to_insert))
                ),
                [n["note_id"] for n in to_insert],
            ).fetchone()[0]
            print(f"{reconnected} existing engram(s) now resolve to a note again.")

    except Exception:
        conn.rollback()
        print("Resync FAILED and was rolled back. Database unchanged.", file=sys.stderr)
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", required=True, type=Path, help="Path to note_registry.db"
    )
    parser.add_argument(
        "--user-id", required=True, help="user_id to assign recovered notes to"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=_DEFAULT_BASE_DIR,
        help=f"study_bubbles directory (default: {_DEFAULT_BASE_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    resync(
        args.db, args.user_id, args.base_dir, dry_run=args.dry_run, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
