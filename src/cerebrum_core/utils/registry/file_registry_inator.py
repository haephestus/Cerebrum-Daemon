import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

from cerebrum_core.utils.file_util_inator import CerebrumPaths


# ==========================================================
# File Registry
# ==========================================================
class FileRegisterInator:
    """
    Registers available files and is the source of truth for which files
    are to be processed and added to domain-specific archives in the knowledgebase.
    """

    def __init__(self, db_path: str = "registry/file_registry.db"):
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
            CREATE TABLE IF NOT EXISTS file_registry (
                id INTEGER PRIMARY KEY,
                file_fingerprint TEXT UNIQUE,
                original_name TEXT,
                sanitized_name TEXT,
                domain TEXT,
                subject TEXT,
                converted INTEGER DEFAULT 0,
                embedded INTEGER DEFAULT 0,
                filepath TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_registry_fingerprint "
            "ON file_registry(file_fingerprint)"
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Register file
    # --------------------------------------------------
    def register_inator(self, original_name: str, filepath: str):
        file_fingerprint = self._file_fingerprint_inator(original_name, filepath)

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO file_registry (
                file_fingerprint,
                original_name,
                filepath
            )
            VALUES (?, ?, ?)
            ON CONFLICT(file_fingerprint) DO UPDATE SET
                last_updated = CURRENT_TIMESTAMP
            """,
            (file_fingerprint, original_name, filepath),
        )

        conn.commit()
        conn.close()

        return file_fingerprint

    # --------------------------------------------------
    # Status updates
    # --------------------------------------------------
    def mark_converted_inator(
        self,
        file_fingerprint: str,
        domain: Optional[str],
        subject: Optional[str],
        sanitized_name: Optional[str],
    ):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE file_registry
            SET
                converted = 1,
                domain = COALESCE(?, domain),
                subject = COALESCE(?, subject),
                sanitized_name = COALESCE(?, sanitized_name),
                last_updated = CURRENT_TIMESTAMP
            WHERE file_fingerprint = ?
            """,
            (domain, subject, sanitized_name, file_fingerprint),
        )

        conn.commit()
        conn.close()

    def mark_embedded_inator(self, file_fingerprint: str):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE file_registry
            SET embedded = 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE file_fingerprint = ?
            """,
            (file_fingerprint,),
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Fetchers
    # --------------------------------------------------
    def fetch_unconverted_file_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                original_name,
                file_fingerprint,
                filepath
            FROM file_registry
            WHERE converted = 0
            """
        )

        rows = cursor.fetchall()
        conn.close()

        columns = ["original_name", "file_fingerprint", "filepath"]
        return [dict(zip(columns, row)) for row in rows]

    def fetch_unembedded_file_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                original_name,
                sanitized_name,
                domain,
                subject,
                file_fingerprint,
                filepath
            FROM file_registry
            WHERE converted = 1 AND embedded = 0
            """
        )

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "original_name",
            "sanitized_name",
            "domain",
            "subject",
            "file_fingerprint",
            "filepath",
        ]
        return [dict(zip(columns, row)) for row in rows]

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def check_inator(self, file_fingerprint: str, field: str = "") -> bool:
        VALID_FIELDS = {"embedded", "converted"}

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        if field:
            if field not in VALID_FIELDS:
                raise ValueError("Invalid field requested")
            cursor.execute(
                f"""
                SELECT {field}
                FROM file_registry
                WHERE file_fingerprint = ?
                """,
                (file_fingerprint,),
            )
        else:
            cursor.execute(
                """
                SELECT 1
                FROM file_registry
                WHERE file_fingerprint = ?
                """,
                (file_fingerprint,),
            )

        result = cursor.fetchone()
        conn.close()

        return bool(result and (result[0] if field else True))

    def show_all_inator(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM file_registry")
        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "file_fingerprint",
            "original_name",
            "sanitized_name",
            "domain",
            "subject",
            "converted",
            "embedded",
            "filepath",
            "last_updated",
        ]

        return [dict(zip(columns, row)) for row in rows]

    # --------------------------------------------------
    # Delete / Reset
    # --------------------------------------------------
    def remove_inator(self, original_name: str, filepath: str, file_fingerprint: str):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                DELETE FROM file_registry
                WHERE original_name = ?
                AND file_fingerprint = ?
                """,
                (original_name, file_fingerprint),
            )

            if cursor.rowcount == 0:
                raise FileNotFoundError("File registry entry not found")

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        path = Path(filepath)
        if path.exists():
            path.unlink()

    def reset_inator(self, status: str, file_fingerprint: Optional[str] = None):
        VALID_COLUMNS = {"embedded", "converted"}
        if status not in VALID_COLUMNS:
            raise ValueError("Invalid status field")

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        if file_fingerprint:
            cursor.execute(
                f"""
                UPDATE file_registry
                SET {status} = 0
                WHERE file_fingerprint = ?
                """,
                (file_fingerprint,),
            )
        else:
            cursor.execute(f"UPDATE file_registry SET {status} = 0")

        conn.commit()
        count = cursor.rowcount
        conn.close()
        return count

    # --------------------------------------------------
    # Fingerprint
    # --------------------------------------------------
    def _file_fingerprint_inator(self, original_name: str, filepath: str) -> str:
        """
        Deterministic fingerprint based on filename + path.
        Prevents collisions across directories.
        """
        payload = f"{original_name}:{filepath}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
