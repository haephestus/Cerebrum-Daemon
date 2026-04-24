import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from cerebrum_core.utils.file_util_inator import CerebrumPaths


# ==========================================================
# Chunk Registry
# ==========================================================
@dataclass
class _NoteChunkRecordInator:
    note_id: str
    chunk_fingerprint: str
    chunk_index: int
    byte_start: int
    byte_end: int
    token_count: int
    chunk_type: str
    parent_chunk_index: Optional[int]
    embedded: int


class NoteChunkRegisterInator:
    def __init__(self, db_path: str = "registry/note_chunk_registry.db"):
        self.db_path = CerebrumPaths().kb_root_dir() / db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS note_chunks(
                id INTEGER PRIMARY KEY,
                note_id TEXT NOT NULL,
                chunk_fingerprint TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                token_count INTEGER,
                chunk_type TEXT NOT NULL,
                parent_chunk_index INTEGER,
                embedded INTEGER DEFAULT 0,
                UNIQUE (note_id, chunk_fingerprint, chunk_index)
            )
            """
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Register chunks
    # --------------------------------------------------
    def register_chunks(self, chunk_rows: List[tuple]):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.executemany(
            """
            INSERT INTO note_chunks(
                note_id,
                chunk_fingerprint,
                chunk_index,
                byte_start,
                byte_end,
                token_count,
                chunk_type,
                parent_chunk_index
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(note_id, chunk_fingerprint, chunk_index)
            DO UPDATE SET
                byte_start = excluded.byte_start,
                byte_end = excluded.byte_end,
                token_count = excluded.token_count,
                chunk_type = excluded.chunk_type,
                parent_chunk_index = excluded.parent_chunk_index
            """,
            chunk_rows,
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Embedding progress
    # --------------------------------------------------
    def get_embedding_progress(self, note_id: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(embedded), 0)
            FROM note_chunks
            WHERE note_id = ?
            """,
            (note_id,),
        )

        total, completed = map(int, cur.fetchone())
        conn.close()

        remaining = total - completed
        progress_pct = (completed / total) * 100 if total > 0 else 0

        return {
            "total": total,
            "completed": completed,
            "remaining": remaining,
            "progress_pct": progress_pct,
        }

    # --------------------------------------------------
    # Chunk updates
    # --------------------------------------------------
    def mark_embedded(self, note_id: str, chunk_fingerprint: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE note_chunks
            SET embedded = 1
            WHERE note_id = ?
              AND chunk_fingerprint = ?
            """,
            (note_id, chunk_fingerprint),
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Fetch unembedded chunks
    # --------------------------------------------------
    def get_unembedded_chunks(self, note_id: str) -> List[_NoteChunkRecordInator]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                note_id,
                chunk_fingerprint,
                chunk_index,
                byte_start,
                byte_end,
                token_count,
                chunk_type,
                parent_chunk_index,
                embedded
            FROM note_chunks
            WHERE note_id = ?
              AND embedded = 0
            ORDER BY chunk_index ASC
            """,
            (note_id,),
        )

        rows = cur.fetchall()
        conn.close()

        return [_NoteChunkRecordInator(*row) for row in rows]

    def show_all_inator(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM note_chunks")
        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "note_id",
            "chunk_fingerprint",
            "chunk_index",
            "byte_start",
            "byte_end",
            "token_count",
            "chunk_type",
            "parent_chunk_index",
            "embedded",
        ]

        return [dict(zip(columns, row)) for row in rows]

    def fetch_chunks_inator(self, note_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                note_id,
                chunk_fingerprint,
                chunk_index,
                byte_start,
                byte_end,
                token_count,
                chunk_type,
                parent_chunk_index
            FROM note_chunks
            WHERE note_id = ?
            """,
            (note_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [_NoteChunkRecordInator(*row) for row in rows]
