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
    pdf_page_start: Optional[int]
    pdf_page_end: Optional[int]
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
                pdf_page_start INTEGER,
                pdf_page_end INTEGER,
                embedded INTEGER DEFAULT 0,
                section_path TEXT,
                section_title TEXT,
                chapter_title TEXT,
                question_number TEXT,
                marks INTEGER,
                zone TEXT,
                UNIQUE (file_fingerprint, chunk_fingerprint, chunk_index)
            )
            """
        )

        # Migration path for existing DBs created before pdf_page_* existed
        cur.execute("PRAGMA table_info(chunks)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "pdf_page_start" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN pdf_page_start INTEGER")
        if "pdf_page_end" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN pdf_page_end INTEGER")
        # Structural breadcrumb columns (TOC-derived). ALTER appends at the
        # end of the table, and the CREATE TABLE above lists them last too,
        # so fresh and migrated DBs keep an identical column order.
        if "section_path" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN section_path TEXT")
        if "section_title" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")
        if "chapter_title" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN chapter_title TEXT")
        # Exam-paper question fields (chunk_type='question'); null otherwise.
        if "question_number" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN question_number TEXT")
        if "marks" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN marks INTEGER")
        # Structural zone (front_matter/body/glossary/index/appendix/...).
        if "zone" not in existing_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN zone TEXT")

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
                parent_chunk_index,
                pdf_page_start,
                pdf_page_end,
                section_path,
                section_title,
                chapter_title,
                question_number,
                marks,
                zone
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_fingerprint, chunk_fingerprint, chunk_index)
            DO UPDATE SET
                byte_start = excluded.byte_start,
                byte_end = excluded.byte_end,
                token_count = excluded.token_count,
                chunk_type = excluded.chunk_type,
                parent_chunk_index = excluded.parent_chunk_index,
                pdf_page_start = excluded.pdf_page_start,
                pdf_page_end = excluded.pdf_page_end,
                section_path = excluded.section_path,
                section_title = excluded.section_title,
                chapter_title = excluded.chapter_title,
                question_number = excluded.question_number,
                marks = excluded.marks,
                zone = excluded.zone
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
    def get_unembedded_chunks(
        self, file_fingerprint: str
    ) -> List[_FileChunkRecordInator]:
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
                pdf_page_start,
                pdf_page_end,
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

    # --------------------------------------------------
    # Fetch a single chunk by position
    # --------------------------------------------------
    def get_chunk(
        self, file_fingerprint: str, chunk_index: int
    ) -> Optional[dict]:
        """
        Resolve a (file_fingerprint, chunk_index) — the identifiers a search
        result / LLM citation carries — to its structural coordinates
        (byte span + PDF page range). Returns None if no such chunk.
        """
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
                pdf_page_start,
                pdf_page_end,
                embedded,
                section_path,
                section_title,
                chapter_title,
                question_number,
                marks,
                zone
            FROM chunks
            WHERE file_fingerprint = ?
              AND chunk_index = ?
            """,
            (file_fingerprint, chunk_index),
        )
        row = cur.fetchone()
        conn.close()

        if row is None:
            return None

        columns = [
            "file_fingerprint",
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
            "section_path",
            "section_title",
            "chapter_title",
            "question_number",
            "marks",
            "zone",
        ]
        return dict(zip(columns, row))

    def show_all_inator(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chunks")
        rows = cursor.fetchall()
        # Derive column names from the cursor rather than a hardcoded list:
        # ALTER-added columns land at the end of the table in an order that
        # varies between fresh and migrated DBs, so a static list can silently
        # misalign. cursor.description always matches the actual row shape.
        columns = [d[0] for d in cursor.description]
        conn.close()

        return [dict(zip(columns, row)) for row in rows]
