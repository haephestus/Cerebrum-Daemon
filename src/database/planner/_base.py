"""
database.study_plan_registry._base
====================================================
Shared plumbing every mixin in this package depends on: connection
handling, the write lock, and schema bootstrap. Deliberately mirrors
cerebrum_core.engrams.storage.note_engram_repository._base — same
short-lived-connection-per-call pattern, same lock discipline — so
anyone who already knows that package doesn't have to learn a second
convention for this one.

Nothing here is plan/phase/week-specific. Domain methods live in the
mixins (plans.py, phases.py, metrics.py, weeks.py) composed onto this
in __init__.py.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Union

from common.file_util_inator import CerebrumPaths

from .schema import SCHEMA_SQL

_DEFAULT_DB_PATH = "registry/study_plan_registry.db"


class _RepositoryBase:
    """
    Connection + schema plumbing only. Holds no domain methods.

    Thread-safety: short-lived connection per call, guarded by a lock
    for writes — matches note_engram_repository's _base.py, since this
    also gets used from a threaded FastAPI app via app.state.
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

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()
