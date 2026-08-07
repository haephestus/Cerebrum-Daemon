import re
import sqlite3

from common.file_util_inator import CerebrumPaths


def concept_slug(text: str) -> str:
    """Normalised identity key for a concept/term — lowercase, punctuation
    collapsed to single hyphens. Two spellings of the same term (casing,
    spacing) resolve to one slug so lookups don't fragment."""
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# ==========================================================
# Concept Index
# ==========================================================
class ConceptIndexInator:
    """
    Concept/definition seeds harvested deterministically from a document's
    structure — glossary term/definition pairs today, printed-index and LLM
    enrichment later. Keyed by file_fingerprint so cleanup mirrors the chunk
    registry. `source` records how a seed was obtained ('glossary' | 'index'
    | 'llm') and is part of the identity so the same term can carry both a
    glossary definition and (later) an LLM-derived one.
    """

    def __init__(self, db_path: str = "registry/concept_index.db"):
        self.db_path = CerebrumPaths().kb_root_dir() / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_index (
                id INTEGER PRIMARY KEY,
                file_fingerprint TEXT NOT NULL,
                concept TEXT NOT NULL,
                concept_slug TEXT NOT NULL,
                definition TEXT,
                source TEXT NOT NULL,
                chunk_index INTEGER,
                pdf_page INTEGER,
                UNIQUE (file_fingerprint, concept_slug, source)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_concept_fp ON concept_index(file_fingerprint)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_concept_slug ON concept_index(concept_slug)"
        )
        conn.commit()
        conn.close()

    def add_concepts(self, rows: list[tuple]):
        """rows: (file_fingerprint, concept, concept_slug, definition,
        source, chunk_index, pdf_page)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO concept_index (
                file_fingerprint, concept, concept_slug, definition,
                source, chunk_index, pdf_page
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_fingerprint, concept_slug, source) DO UPDATE SET
                concept = excluded.concept,
                definition = excluded.definition,
                chunk_index = excluded.chunk_index,
                pdf_page = excluded.pdf_page
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def get_by_fingerprint(self, file_fingerprint: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT concept, concept_slug, definition, source, chunk_index, pdf_page
            FROM concept_index
            WHERE file_fingerprint = ?
            ORDER BY concept_slug
            """,
            (file_fingerprint,),
        )
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def delete_by_fingerprint(self, file_fingerprint: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM concept_index WHERE file_fingerprint = ?",
            (file_fingerprint,),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count
