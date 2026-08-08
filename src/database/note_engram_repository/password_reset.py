"""
database.note_engram_repository.password_reset
==============================================
Storage for the forgotten-password reset flow. One row per reset request; only
a keyed HASH of the emailed shortcode is stored. Creating a new code invalidates
the user's prior outstanding ones; `attempts` locks a code after too many wrong
guesses; `used_at` makes it single-use.
"""

from __future__ import annotations

from typing import Optional

from ._base import _id


class PasswordResetMixin:
    def create_reset_code(
        self, user_id: str, code_hash: str, expires_at: str
    ) -> str:
        """Invalidate any active codes for the user, then insert a fresh one.
        Returns the new reset row id."""
        rid = _id()
        with self._transaction() as conn:
            conn.execute(
                "UPDATE password_reset_codes SET used_at = datetime('now') "
                "WHERE user_id = ? AND used_at IS NULL",
                (user_id,),
            )
            conn.execute(
                "INSERT INTO password_reset_codes (id, user_id, code_hash, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (rid, user_id, code_hash, expires_at),
            )
        return rid

    def get_active_reset(self, user_id: str) -> Optional[dict]:
        """The user's current unused reset row (newest), or None."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM password_reset_codes "
                "WHERE user_id = ? AND used_at IS NULL "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def get_reset(self, reset_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM password_reset_codes WHERE id = ?", (reset_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def increment_reset_attempts(self, reset_id: str) -> int:
        """Bump the wrong-guess counter; return the new value."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE password_reset_codes SET attempts = attempts + 1 WHERE id = ?",
                    (reset_id,),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT attempts FROM password_reset_codes WHERE id = ?",
                    (reset_id,),
                ).fetchone()
                return int(row["attempts"]) if row else 0
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def mark_reset_used(self, reset_id: str) -> None:
        self._simple_write(
            "UPDATE password_reset_codes SET used_at = datetime('now') WHERE id = ?",
            (reset_id,),
        )

    def invalidate_user_resets(self, user_id: str) -> None:
        self._simple_write(
            "UPDATE password_reset_codes SET used_at = datetime('now') "
            "WHERE user_id = ? AND used_at IS NULL",
            (user_id,),
        )

    def _simple_write(self, sql: str, params) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(sql, params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
