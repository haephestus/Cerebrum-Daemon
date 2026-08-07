import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

from common.file_util_inator import CerebrumPaths


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
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                doc_type TEXT DEFAULT 'unknown'
            )
            """
        )

        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_registry_fingerprint "
            "ON file_registry(file_fingerprint)"
        )

        # Discretionary access grants. A file with NO rows here is PUBLIC (the
        # default — every existing file stays pooled/shared, nothing to
        # backfill). Once it has one or more rows it's PRIVATE, visible only to
        # the listed principals. principal_type is 'user' or 'org'; principal_id
        # is a users(id) or orgs(id) from the *other* DB (note_registry.db), so
        # there's no cross-DB FK — a stale grant simply matches no one.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_access (
                file_fingerprint TEXT NOT NULL,
                principal_type   TEXT NOT NULL CHECK(principal_type IN ('user','org')),
                principal_id     TEXT NOT NULL,
                PRIMARY KEY (file_fingerprint, principal_type, principal_id)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_access_fp "
            "ON file_access(file_fingerprint)"
        )

        # Migration for DBs created before doc_type existed. ALTER appends at
        # the end, matching the CREATE TABLE above (doc_type listed last), so
        # fresh and migrated DBs keep an identical column order.
        cursor.execute("PRAGMA table_info(file_registry)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "doc_type" not in existing_cols:
            cursor.execute(
                "ALTER TABLE file_registry ADD COLUMN doc_type TEXT DEFAULT 'unknown'"
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
        doc_type: Optional[str] = None,
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
                doc_type = COALESCE(?, doc_type),
                last_updated = CURRENT_TIMESTAMP
            WHERE file_fingerprint = ?
            """,
            (domain, subject, sanitized_name, doc_type, file_fingerprint),
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

    def get_by_fingerprint(self, file_fingerprint: str) -> Optional[dict]:
        """
        Fetch a single registry row by fingerprint. Returns None if not found.
        """
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id, file_fingerprint, original_name, sanitized_name,
                domain, subject, converted, embedded, filepath, last_updated,
                doc_type
            FROM file_registry
            WHERE file_fingerprint = ?
            """,
            (file_fingerprint,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

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
            "doc_type",
        ]
        return dict(zip(columns, row))

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

    def show_all_inator(self, user_id: Optional[str] = None, org_ids=None):
        """List registry entries. With no args, returns everything (unchanged
        legacy behaviour). Pass user_id (and optionally the user's org_ids) to
        get only the files visible to that principal: all public files plus any
        privately granted to the user or one of their orgs."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM file_registry")
        rows = cursor.fetchall()
        # Derive columns from the cursor so ALTER-added columns (doc_type, and
        # any future ones) appear without a hardcoded list silently truncating
        # or misaligning them.
        columns = [d[0] for d in cursor.description]
        conn.close()

        records = [dict(zip(columns, row)) for row in rows]
        if user_id is None:
            return records

        visible = set(
            self.filter_visible(
                [r["file_fingerprint"] for r in records], user_id, org_ids
            )
        )
        return [r for r in records if r["file_fingerprint"] in visible]

    # --------------------------------------------------
    # Discretionary access (file_access)
    # --------------------------------------------------
    def grant_access(
        self, file_fingerprint: str, principal_type: str, principal_id: str
    ):
        """Grant a user/org access to a file. The first grant on a file flips
        it from public to private — after this, only granted principals (and
        anyone with a later grant) can see it."""
        if principal_type not in ("user", "org"):
            raise ValueError("principal_type must be 'user' or 'org'")
        conn = sqlite3.connect(self.DB_PATH)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO file_access "
                "(file_fingerprint, principal_type, principal_id) VALUES (?, ?, ?)",
                (file_fingerprint, principal_type, principal_id),
            )
            conn.commit()
        finally:
            conn.close()

    def revoke_access(
        self, file_fingerprint: str, principal_type: str, principal_id: str
    ) -> bool:
        """Revoke one grant. Returns True if a grant was removed. Removing the
        last grant makes the file public again."""
        conn = sqlite3.connect(self.DB_PATH)
        try:
            cur = conn.execute(
                "DELETE FROM file_access WHERE file_fingerprint = ? "
                "AND principal_type = ? AND principal_id = ?",
                (file_fingerprint, principal_type, principal_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_access(self, file_fingerprint: str) -> list:
        """The grants on a file. Empty list means the file is public."""
        conn = sqlite3.connect(self.DB_PATH)
        try:
            rows = conn.execute(
                "SELECT principal_type, principal_id FROM file_access "
                "WHERE file_fingerprint = ?",
                (file_fingerprint,),
            ).fetchall()
        finally:
            conn.close()
        return [{"principal_type": t, "principal_id": i} for t, i in rows]

    def filter_visible(self, fingerprints, user_id: str, org_ids=None) -> list:
        """Given candidate fingerprints (e.g. from a vector search or the full
        registry), return those the principal may see: public files (no grants)
        plus files granted to this user or one of their orgs. Order preserved.

        This is the single enforcement point for #3 — routes call it rather
        than reimplementing the rule, and it touches no chunk/vector metadata.
        """
        fingerprints = list(fingerprints)
        if not fingerprints:
            return []

        conn = sqlite3.connect(self.DB_PATH)
        try:
            restricted = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT file_fingerprint FROM file_access"
                ).fetchall()
            }
            if not restricted:
                return fingerprints  # nothing is scoped — everything is public

            principals = [("user", user_id)]
            principals += [("org", o) for o in (org_ids or [])]
            clause = " OR ".join(
                ["(principal_type = ? AND principal_id = ?)"] * len(principals)
            )
            params = [v for pair in principals for v in pair]
            shared = {
                r[0]
                for r in conn.execute(
                    f"SELECT file_fingerprint FROM file_access WHERE {clause}", params
                ).fetchall()
            }
        finally:
            conn.close()

        return [fp for fp in fingerprints if fp not in restricted or fp in shared]

    # --------------------------------------------------
    # Delete / Reset
    # --------------------------------------------------
    def remove_inator(self, file_fingerprint: str):
        """
        Remove a file's registry entry and its file on disk.
        Looks up filepath from the registry itself — the caller only
        needs to know the fingerprint.
        """
        row = self.get_by_fingerprint(file_fingerprint)
        if row is None:
            raise FileNotFoundError(
                f"No registry entry found for fingerprint {file_fingerprint}"
            )

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM file_registry WHERE file_fingerprint = ?",
                (file_fingerprint,),
            )

            if cursor.rowcount == 0:
                raise FileNotFoundError("File registry entry not found")

            # No cross-DB FK/cascade here, so drop this file's grants explicitly
            # to avoid leaving orphaned rows in file_access.
            cursor.execute(
                "DELETE FROM file_access WHERE file_fingerprint = ?",
                (file_fingerprint,),
            )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        path = Path(row["filepath"])
        if path.exists():
            path.unlink()

        return row

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
