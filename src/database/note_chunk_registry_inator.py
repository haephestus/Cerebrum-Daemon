import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from common.file_util_inator import CerebrumPaths


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

        # NOTE chunks — distinct from the file-chunk table (`chunks`, owned by
        # FileChunkRegisterInator for source-file ingestion). Every method here
        # targets note_chunks; the old copy-pasted `chunks`/file_fingerprint
        # CREATE never matched the writer, so this table wasn't actually created.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS note_chunks (
                id INTEGER PRIMARY KEY,
                note_id TEXT NOT NULL,
                chunk_fingerprint TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                token_count INTEGER,
                chunk_type TEXT NOT NULL,
                parent_chunk_index INTEGER,
                pdf_page_start INTEGER,
                pdf_page_end INTEGER,
                embedded INTEGER DEFAULT 0,
                UNIQUE (note_id, chunk_fingerprint, chunk_index)
            )
            """
        )

        # chunk ↔ block join (gap 1 / stream A). Replaces source_block_ids-in-a-
        # comment: which blocks each chunk owns, queryable BOTH directions
        # (chunk→blocks for highlighting; block→chunk for tap-to-analysis).
        # is_partial marks a piece of an oversized split block.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS note_chunk_blocks (
                note_id     TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                block_id    TEXT NOT NULL,
                ordinal     INTEGER NOT NULL,
                is_partial  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (note_id, chunk_index, block_id)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ncb_chunk ON note_chunk_blocks(note_id, chunk_index)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ncb_block ON note_chunk_blocks(note_id, block_id)"
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Register chunks
    # --------------------------------------------------
    def register_chunks(self, chunk_rows: List[tuple]):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Delete existing rows for this note so re-chunking replaces stale offsets
        note_id = chunk_rows[0][0] if chunk_rows else None
        if note_id:
            cur.execute("DELETE FROM note_chunks WHERE note_id = ?", (note_id,))

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
                parent_chunk_index,
                pdf_page_start,
                pdf_page_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ? ,?)
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
            "pdf_page_start",
            "pdf_page_end",
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
                parent_chunk_index,
                embedded
            FROM note_chunks
            WHERE note_id = ?
            """,
            (note_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [_NoteChunkRecordInator(*row) for row in rows]

    # --------------------------------------------------
    # Chunk ↔ block join (gap 1 / stream A)
    # --------------------------------------------------
    def register_chunk_blocks(self, note_id: str, rows: List[tuple]):
        """Replace a note's chunk↔block rows. Each row is
        (note_id, chunk_index, block_id, ordinal, is_partial)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM note_chunk_blocks WHERE note_id = ?", (note_id,))
        if rows:
            cur.executemany(
                """
                INSERT OR REPLACE INTO note_chunk_blocks
                    (note_id, chunk_index, block_id, ordinal, is_partial)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()
        conn.close()

    def blocks_for_chunk(self, note_id: str, chunk_index: int) -> List[dict]:
        """The blocks a chunk owns (for highlighting a weak chunk's span)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT block_id, ordinal, is_partial FROM note_chunk_blocks
            WHERE note_id = ? AND chunk_index = ? ORDER BY ordinal
            """,
            (note_id, chunk_index),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {"block_id": b, "ordinal": o, "is_partial": bool(p)} for b, o, p in rows
        ]

    def chunks_for_block(self, note_id: str, block_id: str) -> List[int]:
        """The chunk(s) covering a block (for tap-a-block → its analysis)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk_index FROM note_chunk_blocks
            WHERE note_id = ? AND block_id = ? ORDER BY chunk_index
            """,
            (note_id, block_id),
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
