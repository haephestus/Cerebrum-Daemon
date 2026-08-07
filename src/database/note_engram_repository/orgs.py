"""
database.note_engram_repository.orgs
==================================================
Organisations and membership. Orgs group users so a shared resource (today:
knowledgebase files, via the file_registry's file_access table) can be scoped
to a set of people rather than a single owner.

This mixin owns the `orgs` and `org_members` tables. It deliberately knows
nothing about files — the file registry lives in a separate SQLite DB and can't
JOIN across to here, so membership is resolved *here* (get_user_org_ids) and the
resulting principal list is handed to the file registry for filtering.
"""

from __future__ import annotations

from typing import Optional


class OrgsMixin:
    def create_org(self, org_id: str, name: str, owner_user_id: str) -> None:
        """Create an org and enrol its creator as 'owner', atomically.
        No-op on the org row if the id already exists (matches the
        create-if-missing pattern used elsewhere); the owner membership is
        upserted so a re-run doesn't demote an existing role."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO orgs (id, name) VALUES (?, ?) "
                    "ON CONFLICT(id) DO NOTHING",
                    (org_id, name),
                )
                conn.execute(
                    "INSERT INTO org_members (org_id, user_id, role) "
                    "VALUES (?, ?, 'owner') "
                    "ON CONFLICT(org_id, user_id) DO NOTHING",
                    (org_id, owner_user_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_org(self, org_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM orgs WHERE id = ?", (org_id,)).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def add_member(self, org_id: str, user_id: str, role: str = "member") -> None:
        """Add (or update the role of) a member. Upsert so calling twice is
        safe and can be used to promote/demote."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO org_members (org_id, user_id, role) VALUES (?, ?, ?) "
                    "ON CONFLICT(org_id, user_id) DO UPDATE SET role = excluded.role",
                    (org_id, user_id, role),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def remove_member(self, org_id: str, user_id: str) -> bool:
        """Remove a member. Returns True if a row was actually removed."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM org_members WHERE org_id = ? AND user_id = ?",
                    (org_id, user_id),
                )
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def list_members(self, org_id: str) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT user_id, role, joined_at FROM org_members "
                "WHERE org_id = ? ORDER BY joined_at",
                (org_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def get_user_org_ids(self, user_id: str) -> list[str]:
        """Every org this user belongs to. This is the bridge to the file
        registry: the caller passes the returned ids as 'org' principals when
        asking which files are visible to the user."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT org_id FROM org_members WHERE user_id = ?", (user_id,)
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def get_member_role(self, org_id: str, user_id: str) -> Optional[str]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT role FROM org_members WHERE org_id = ? AND user_id = ?",
                (org_id, user_id),
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def delete_org(self, org_id: str) -> bool:
        """Delete an org; org_members rows cascade. Returns True if it existed.
        Note: this does NOT touch file_access grants that reference this org id
        (a different DB) — the file registry treats a dangling org grant as
        simply matching no one, so it's harmless, but callers may want to
        revoke org grants explicitly first."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("DELETE FROM orgs WHERE id = ?", (org_id,))
                conn.commit()
                return cur.rowcount > 0
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
