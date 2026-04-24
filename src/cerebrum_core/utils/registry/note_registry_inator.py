import sqlite3
from pathlib import Path
from typing import Optional

from cerebrum_core.utils.file_util_inator import CerebrumPaths


# ==========================================================
# Note Registry
# ==========================================================
class NoteRegisterInator:
    """
    Registers notes and tracks their analysis / caching state
    for the knowledgebase.
    """

    def __init__(self, db_path: str = "registry/note_registry.db"):
        self.DB_PATH = CerebrumPaths().kb_root_dir() / db_path
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._table_initiator_inator()

    # --------------------------------------------------
    # Table setup
    # --------------------------------------------------
    def _table_initiator_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS note_registry (
                id INTEGER PRIMARY KEY,
                bubble_id TEXT UNIQUE,
                note_id TEXT UNIQUE NOT NULL,
                domain TEXT,
                subject TEXT,
                cached INTEGER DEFAULT 0,
                analysed INTEGER DEFAULT 0,
                filepath TEXT,
                last_analysed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_note_registry_note_id
            ON note_registry(note_id)
            """
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Register note
    # --------------------------------------------------
    def register_inator(
        self,
        note_id: str,
        bubble_id: Optional[str],
        filepath: str,
    ):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO note_registry (
                note_id,
                bubble_id,
                filepath
            )
            VALUES (?, ?, ?)
            ON CONFLICT(note_id) DO UPDATE SET
                last_analysed = CURRENT_TIMESTAMP
            """,
            (note_id, bubble_id, filepath),
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Status updates
    # --------------------------------------------------
    def mark_cached_inator(self, note_id: str):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE note_registry
            SET cached = 1,
                last_analysed = CURRENT_TIMESTAMP
            WHERE note_id = ?
            """,
            (note_id,),
        )

        conn.commit()
        conn.close()

    def mark_analysed_inator(
        self,
        note_id: str,
        domain: Optional[str] = "",
        subject: Optional[str] = "",
    ):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE note_registry
            SET
                analysed = 1,
                domain = COALESCE(?, domain),
                subject = COALESCE(?, subject),
                last_analysed = CURRENT_TIMESTAMP
            WHERE note_id = ?
            """,
            (domain, subject, note_id),
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Fetchers
    # --------------------------------------------------
    def fetch_uncached_notes_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT note_id, bubble_id, filepath
            FROM note_registry
            WHERE cached = 0
            """
        )

        rows = cursor.fetchall()
        conn.close()

        columns = ["note_id", "bubble_id", "filepath"]
        return [dict(zip(columns, row)) for row in rows]

    def fetch_unanalysed_notes_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                note_id,
                bubble_id,
                domain,
                subject,
                filepath
            FROM note_registry
            WHERE analysed = 0
            """
        )

        rows = cursor.fetchall()
        conn.close()

        columns = ["note_id", "bubble_id", "domain", "subject", "filepath"]
        return [dict(zip(columns, row)) for row in rows]

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def check_inator(self, note_id: str, field: str = "") -> bool:
        VALID_FIELDS = {"cached", "analysed"}

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        if field:
            if field not in VALID_FIELDS:
                raise ValueError("Invalid field requested")
            cursor.execute(
                f"""
                SELECT {field}
                FROM note_registry
                WHERE note_id = ?
                """,
                (note_id,),
            )
        else:
            cursor.execute(
                """
                SELECT 1
                FROM note_registry
                WHERE note_id = ?
                """,
                (note_id,),
            )

        result = cursor.fetchone()
        conn.close()

        return bool(result and (result[0] if field else True))

    def show_all_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM note_registry")
        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "bubble_id",
            "note_id",
            "domain",
            "subject",
            "cached",
            "analysed",
            "filepath",
            "last_analysed",
        ]

        return [dict(zip(columns, row)) for row in rows]

    # --------------------------------------------------
    # Delete / Reset
    # --------------------------------------------------
    def remove_inator(self, note_id: str, filepath: str):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                DELETE FROM note_registry
                WHERE note_id = ?
                """,
                (note_id,),
            )

            if cursor.rowcount == 0:
                raise FileNotFoundError("Note registry entry not found")

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        path = Path(filepath)
        if path.exists():
            path.unlink()

    def reset_inator(self, status: str, note_id: Optional[str] = None):
        VALID_COLUMNS = {"cached", "analysed"}
        if status not in VALID_COLUMNS:
            raise ValueError("Invalid status field")

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        if note_id:
            cursor.execute(
                f"""
                UPDATE note_registry
                SET {status} = 0
                WHERE note_id = ?
                """,
                (note_id,),
            )
        else:
            cursor.execute(f"UPDATE note_registry SET {status} = 0")

        conn.commit()
        count = cursor.rowcount
        conn.close()
        return count
