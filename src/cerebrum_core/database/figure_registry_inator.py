import sqlite3

from cerebrum_core.utils.file_util_inator import CerebrumPaths


# ==========================================================
# Figure Registry
# ==========================================================
class FigureRegisterInator:
    """
    Figures extracted from a source PDF, stored as first-class objects rather
    than shoehorned into the byte-based text-chunk registry (an image has no
    meaningful byte span in the markdown). Each row holds the page + bounding
    box needed to render a crop on demand and the nearby caption used as the
    LLM-facing anchor. `description` is reserved for the later vision-model
    upgrade (a textual render of the figure) and is null in the baseline.
    """

    def __init__(self, db_path: str = "registry/figure_registry.db"):
        self.db_path = CerebrumPaths().kb_root_dir() / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS figures (
                id INTEGER PRIMARY KEY,
                file_fingerprint TEXT NOT NULL,
                figure_index INTEGER NOT NULL,
                pdf_page INTEGER NOT NULL,
                bbox TEXT NOT NULL,
                caption TEXT,
                description TEXT,
                UNIQUE (file_fingerprint, figure_index)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_figures_fp ON figures(file_fingerprint)"
        )
        conn.commit()
        conn.close()

    def register_figures(self, rows: list[tuple]):
        """rows: (file_fingerprint, figure_index, pdf_page, bbox_json, caption)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO figures (
                file_fingerprint, figure_index, pdf_page, bbox, caption
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_fingerprint, figure_index) DO UPDATE SET
                pdf_page = excluded.pdf_page,
                bbox = excluded.bbox,
                caption = excluded.caption
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def get_figure(self, file_fingerprint: str, figure_index: int):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT file_fingerprint, figure_index, pdf_page, bbox, caption, description
            FROM figures
            WHERE file_fingerprint = ? AND figure_index = ?
            """,
            (file_fingerprint, figure_index),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        columns = [
            "file_fingerprint",
            "figure_index",
            "pdf_page",
            "bbox",
            "caption",
            "description",
        ]
        return dict(zip(columns, row))

    def list_figures(self, file_fingerprint: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT figure_index, pdf_page, bbox, caption, description
            FROM figures
            WHERE file_fingerprint = ?
            ORDER BY figure_index
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
            "DELETE FROM figures WHERE file_fingerprint = ?",
            (file_fingerprint,),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count
