"""
cerebrum_core.engrams.storage.note_engram_repository._base
============================================================
Shared plumbing every mixin in this package depends on: connection
handling, the write lock, schema bootstrap, and small id/time helpers.

Nothing here is domain-specific (no notes/engrams/mastery knowledge) —
that's the whole point of splitting it out. Every mixin assumes it's
being composed onto a class that also inherits _RepositoryBase, so it
can call self._get_conn() / self._lock without owning either.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union

from cerebrum_core.utils.file_util_inator import CerebrumPaths

from .schema import SCHEMA_SQL

_DEFAULT_DB_PATH = "registry/note_registry.db"


def _now() -> str:
    return datetime.utcnow().isoformat()


def _id() -> str:
    return uuid.uuid4().hex


class _RepositoryBase:
    """
    Connection + schema plumbing only. Holds no domain methods — those
    live in the mixins that get composed alongside this in
    NoteEngramRepository (see __init__.py).

    Thread-safety: unlike a single held connection, this opens a
    short-lived connection per call, guarded by a lock for writes —
    matching the original NoteRegisterInator's pattern, since this is
    what gets used from a threaded FastAPI app via app.state.
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
    def open(cls, db_path: Union[str, Path, None] = None) -> "_RepositoryBase":
        return cls(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.DB_PATH, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run several statements as one atomic unit under the write lock.

        Yields a single connection; commits once on clean exit, rolls back on
        any exception. Use this for logical operations that span multiple
        tables (e.g. attempt + responses + grading job) so a partial failure
        can't leave orphaned rows — the per-call helpers each open their own
        connection and commit independently, which is fine for single writes
        but not for multi-step ones.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Schema setup + migration
    # -----------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
                # SCHEMA_SQL (CREATE IF NOT EXISTS) covers fresh DBs; migrations
                # bring existing DBs whose shape predates a change up to date.
                from .migrations import run_migrations

                run_migrations(conn)
            finally:
                conn.close()

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def _columns_of(self, conn: sqlite3.Connection, table: str) -> list[str]:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
