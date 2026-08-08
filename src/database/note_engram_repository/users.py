"""
cerebrum_core.engrams.storage.note_engram_repository.users
==============================================================
User registration/lookup. Small enough to earn its own file rather than
being folded into notes.py or attempts.py -- it's referenced by both
(engram_attempts.user_id, engram_mastery.user_id) but owns neither.
"""

from __future__ import annotations

import json
from typing import Optional


class UsersMixin:
    def create_user(
        self,
        user_id: str,
        name: str,
        email: str,
        password_hash: str,
        settings: Optional[dict] = None,
    ) -> None:
        """
        Register a new user, or no-op if the id already exists.
        `password_hash` is an opaque bcrypt string produced at the API edge
        (api.auth.password_inator) — this layer never sees the plaintext.
        Note: ON CONFLICT(id) DO NOTHING means calling this twice with the
        same id but different fields won't update anything -- matches the
        pattern used elsewhere here (e.g. register_inator) of "create if
        missing" rather than upsert. A duplicate *email* still raises
        sqlite3.IntegrityError (email is UNIQUE), which the route maps to 409.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO users (id, name, email, password_hash, settings)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        user_id,
                        name,
                        email,
                        password_hash,
                        json.dumps(settings or {}),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_user(self, user_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Look up a user by their (unique) email, including password_hash.
        Used only by the login route to verify credentials — callers must not
        return the raw dict to clients, since it carries the hash."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def update_password(self, user_id: str, password_hash: str) -> bool:
        """Set a user's bcrypt password hash. Returns True if a row was updated
        (False if user_id doesn't exist). `password_hash` is opaque here —
        hashing happens at the API edge (api.auth.password_inator)."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (password_hash, user_id),
                )
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def delete_user(self, user_id: str) -> bool:
        """
        Permanently delete a user and everything that belongs to them.

        Most tables here reference users(id) WITHOUT ON DELETE CASCADE
        (only notes -> engrams -> {mcq,flashcard,long_question}_content /
        long_question_parts / short_question cascade automatically), and
        foreign_keys=ON means a bare DELETE FROM users would just throw
        FOREIGN KEY constraint failed the moment the user has any data.
        So we walk dependents in FK-safe order, inside one transaction:

          1. rows that reference engram_attempts.id (the *_responses
             tables + grading_jobs) for this user's attempts
          2. engram_attempts itself (by user_id)
          3. engram_mastery, topic_mastery, misconceptions,
             engram_generation_queue (by user_id)
          4. notes (by user_id) -- cascades automatically down through
             engrams -> content tables per the schema's ON DELETE CASCADE
          5. the user row itself

        Returns True if a user was actually deleted, False if user_id
        didn't exist.

        CAVEAT: step 4 assumes notes are single-owner (per the schema
        comment: "owner; every note belongs to exactly one user"). If
        another user somehow has an engram_attempts row pointing at an
        engram that belongs to *this* user's note, that FK isn't cleaned
        up here and the cascade will fail. Not expected to happen given
        the current ownership model, but worth knowing if bubbles/notes
        ever become shared across users.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                if (
                    conn.execute(
                        "SELECT 1 FROM users WHERE id = ?", (user_id,)
                    ).fetchone()
                    is None
                ):
                    return False

                attempt_ids = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM engram_attempts WHERE user_id = ?",
                        (user_id,),
                    ).fetchall()
                ]

                if attempt_ids:
                    placeholders = ",".join("?" * len(attempt_ids))
                    for table in (
                        "mcq_responses",
                        "flashcard_responses",
                        "short_question_responses",
                        "long_question_responses",
                        "grading_jobs",
                    ):
                        conn.execute(
                            f"DELETE FROM {table} WHERE attempt_id IN ({placeholders})",
                            attempt_ids,
                        )

                conn.execute(
                    "DELETE FROM engram_attempts WHERE user_id = ?", (user_id,)
                )
                conn.execute("DELETE FROM engram_mastery WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM topic_mastery WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM misconceptions WHERE user_id = ?", (user_id,))
                conn.execute(
                    "DELETE FROM engram_generation_queue WHERE user_id = ?",
                    (user_id,),
                )

                # Cascades to engrams -> mcq_content/flashcard_content/
                # long_question_content/long_question_parts/short_question
                conn.execute("DELETE FROM notes WHERE user_id = ?", (user_id,))

                conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
