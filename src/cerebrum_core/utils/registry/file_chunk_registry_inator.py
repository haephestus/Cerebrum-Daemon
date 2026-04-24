import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from cerebrum_core.utils.file_util_inator import CerebrumPaths


# ==========================================================
# Chunk Registry
# ==========================================================
@dataclass
class _FileChunkRecordInator:
    file_fingerprint: str
    chunk_fingerprint: str
    chunk_index: int
    byte_start: int
    byte_end: int
    token_count: int
    chunk_type: str
    parent_chunk_index: Optional[int]
    embedded: int


class FileChunkRegisterInator:
    def __init__(self, db_path: str = "registry/file_chunk_registry.db"):
        self.db_path = CerebrumPaths().kb_root_dir() / db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_fingerprint TEXT NOT NULL,
                chunk_fingerprint TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                token_count INTEGER,
                chunk_type TEXT NOT NULL,
                parent_chunk_index INTEGER,
                embedded INTEGER DEFAULT 0,
                UNIQUE (file_fingerprint, chunk_fingerprint, chunk_index)
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
            INSERT INTO chunks (
                file_fingerprint,
                chunk_fingerprint,
                chunk_index,
                byte_start,
                byte_end,
                token_count,
                chunk_type,
                parent_chunk_index
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_fingerprint, chunk_fingerprint, chunk_index)
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
    def get_embedding_progress(self, file_fingerprint: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(embedded), 0)
            FROM chunks
            WHERE file_fingerprint = ?
            """,
            (file_fingerprint,),
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
    def mark_embedded(self, file_fingerprint: str, chunk_fingerprint: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE chunks
            SET embedded = 1
            WHERE file_fingerprint = ?
              AND chunk_fingerprint = ?
            """,
            (file_fingerprint, chunk_fingerprint),
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Fetch unembedded chunks
    # --------------------------------------------------
    def get_unembedded_chunks(self, file_fingerprint: str) -> List[_FileChunkRecordInator]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                file_fingerprint,
                chunk_fingerprint,
                chunk_index,
                byte_start,
                byte_end,
                token_count,
                chunk_type,
                parent_chunk_index,
                embedded
            FROM chunks
            WHERE file_fingerprint = ?
              AND embedded = 0
            ORDER BY chunk_index ASC
            """,
            (file_fingerprint,),
        )

        rows = cur.fetchall()
        conn.close()

        return [_FileChunkRecordInator(*row) for row in rows]

    def show_all_inator(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chunks")
        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "file_fingerprint",
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
