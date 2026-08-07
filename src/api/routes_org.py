"""
api.routes_org
==============
Organisation management: create orgs, manage membership, and read who belongs.
Orgs exist to scope shared resources (today: knowledgebase files, via the file
registry's file_access grants) to a group instead of a single owner.

Every route derives the caller from X-User-Id (get_current_user_id). Mutations
are gated on the caller's role in the org — 'owner'/'admin' may manage members,
'owner' may delete the org, and any member may read.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from cerebrum_core.utils.user_context_inator import get_current_user_id

org_router = APIRouter(prefix="/org", tags=["orgs"])

_MANAGE_ROLES = {"owner", "admin"}


class CreateOrgRequest(BaseModel):
    name: str


class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "member"


def _require_role(repo, org_id: str, user_id: str, allowed: set) -> str:
    """Return the caller's role in the org, or 404/403 if they can't act.
    404 (not 403) when the caller isn't a member at all, so org existence
    isn't leaked to outsiders."""
    role = repo.get_member_role(org_id, user_id)
    if role is None:
        raise HTTPException(status_code=404, detail=f"Org not found: {org_id}")
    if role not in allowed:
        raise HTTPException(
            status_code=403, detail="Insufficient role for this action"
        )
    return role


@org_router.post("")
def create_org(
    body: CreateOrgRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    org_id = uuid.uuid4().hex
    try:
        repo.create_org(org_id=org_id, name=body.name, owner_user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"id": org_id, "name": body.name, "owner": user_id}


@org_router.get("")
def list_my_orgs(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    repo = request.app.state.note_registry
    orgs = [repo.get_org(oid) for oid in repo.get_user_org_ids(user_id)]
    return {"orgs": [o for o in orgs if o]}


@org_router.get("/{org_id}")
def get_org(
    org_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    repo = request.app.state.note_registry
    _require_role(repo, org_id, user_id, allowed={"owner", "admin", "member"})
    return repo.get_org(org_id)


@org_router.delete("/{org_id}")
def delete_org(
    org_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    repo = request.app.state.note_registry
    _require_role(repo, org_id, user_id, allowed={"owner"})
    deleted = repo.delete_org(org_id)
    return {"deleted": deleted, "id": org_id}


@org_router.get("/{org_id}/members")
def list_members(
    org_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    repo = request.app.state.note_registry
    _require_role(repo, org_id, user_id, allowed={"owner", "admin", "member"})
    return {"org_id": org_id, "members": repo.list_members(org_id)}


@org_router.post("/{org_id}/members")
def add_member(
    org_id: str,
    body: AddMemberRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    _require_role(repo, org_id, user_id, allowed=_MANAGE_ROLES)
    if body.role not in ("owner", "admin", "member"):
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.role}")
    if not repo.get_user(body.user_id):
        raise HTTPException(status_code=404, detail=f"Unknown user: {body.user_id}")
    repo.add_member(org_id, body.user_id, body.role)
    return {"org_id": org_id, "user_id": body.user_id, "role": body.role}


@org_router.delete("/{org_id}/members/{member_id}")
def remove_member(
    org_id: str,
    member_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    # Managers may remove anyone; any member may remove themselves (leave).
    if member_id != user_id:
        _require_role(repo, org_id, user_id, allowed=_MANAGE_ROLES)
    else:
        _require_role(repo, org_id, user_id, allowed={"owner", "admin", "member"})
    removed = repo.remove_member(org_id, member_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Membership not found")
    return {"org_id": org_id, "removed": member_id}
