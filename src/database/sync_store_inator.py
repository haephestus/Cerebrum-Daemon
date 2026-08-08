"""
database.sync_store_inator — per-node local sync bookkeeping (gap 1 / stream C).

NOT synced content — each node keeps its own. Three tables:

  replica_identity  — this node's stable id (the slot name it owns in every
                      version vector). Minted once; must survive restarts.
  sync_outbox       — local edits not yet acked by a hub. While offline every
                      edit lands here; on reconnect we drain it. Deleted on ack.
  sync_cursor       — per-peer "where we left off": the version vector last fully
                      exchanged with that peer, so "give me everything since X"
                      is well-defined (a device talks to >1 hub: LAN daemon +
                      cloud, each needs its own cursor).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Optional

from common.file_util_inator import CerebrumPaths


class SyncStoreInator:
    def __init__(self, db_path: str = "registry/sync_store.db"):
        self.db_path = CerebrumPaths().kb_root_dir() / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS replica_identity (
                    replica_id TEXT PRIMARY KEY,
                    role       TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS sync_outbox (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id    TEXT NOT NULL,
                    page_id    TEXT,
                    kind       TEXT NOT NULL,   -- 'content' | 'ink' | 'meta' | 'tombstone'
                    payload    TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_note ON sync_outbox(note_id);
                CREATE TABLE IF NOT EXISTS sync_cursor (
                    peer_id        TEXT PRIMARY KEY,
                    last_vector    TEXT NOT NULL DEFAULT '{}',
                    last_synced_at TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    # -- replica identity --------------------------------------------------
    def get_or_create_replica_id(self, role: str = "node") -> str:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT replica_id FROM replica_identity LIMIT 1"
            ).fetchone()
            if row:
                return row["replica_id"]
            rid = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO replica_identity (replica_id, role) VALUES (?, ?)",
                (rid, role),
            )
            conn.commit()
            return rid
        finally:
            conn.close()

    # -- outbox ------------------------------------------------------------
    def enqueue(self, note_id: str, kind: str, payload: dict, page_id: Optional[str] = None) -> int:
        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT INTO sync_outbox (note_id, page_id, kind, payload) VALUES (?, ?, ?, ?)",
                (note_id, page_id, kind, json.dumps(payload)),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def drain(self, limit: int = 100) -> list[dict]:
        """Oldest pending edits (peek — rows stay until acked)."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM sync_outbox ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def ack(self, ids: list[int]) -> None:
        if not ids:
            return
        conn = self._conn()
        try:
            conn.executemany("DELETE FROM sync_outbox WHERE id = ?", [(i,) for i in ids])
            conn.commit()
        finally:
            conn.close()

    def outbox_size(self) -> int:
        conn = self._conn()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM sync_outbox").fetchone()[0])
        finally:
            conn.close()

    # -- per-peer cursor ---------------------------------------------------
    def get_cursor(self, peer_id: str) -> dict:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT last_vector, last_synced_at FROM sync_cursor WHERE peer_id = ?",
                (peer_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {"last_vector": {}, "last_synced_at": None}
        return {
            "last_vector": json.loads(row["last_vector"] or "{}"),
            "last_synced_at": row["last_synced_at"],
        }

    def set_cursor(self, peer_id: str, last_vector: dict) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO sync_cursor (peer_id, last_vector, last_synced_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(peer_id) DO UPDATE SET
                    last_vector = excluded.last_vector,
                    last_synced_at = excluded.last_synced_at
                """,
                (peer_id, json.dumps(last_vector)),
            )
            conn.commit()
        finally:
            conn.close()
