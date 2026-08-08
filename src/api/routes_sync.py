"""
api.routes_sync — note offline-sync endpoints (gap 1 / stream C).

The hub side of the protocol the client's outbox drains into:

  POST /sync/push/{bubble_id}/{note_id}  — client sends its note version; the
        server MERGES it into the stored one (version-vector: dominates→take
        newer, concurrent→LWW-per-page, ink union) and returns the merged note +
        which pages conflicted + the merged vector.
  GET  /sync/pull/{bubble_id}/{note_id}  — the stored note + its vector, so a
        client can fast-forward. (Pull-since-cursor across many notes needs
        per-note change tracking — deferred.)
  GET  /sync/replica-id                  — the hub's stable replica id (the peer
        id a client keys its sync cursor on).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

# Reuse the SAME owner check the note routes use (it reads <bubble>/info.json).
# A local reimpl here read the bubble *directory* as JSON and 404'd on every
# request — so if a note saves via /bubbles it must also sync via /sync.
from api.routes_bubble import _assert_bubble_owner
from cerebrum_core.user_context_inator import get_current_user_id
from common.file_util_inator import CerebrumPaths
from database.sync_store_inator import SyncStoreInator
from models.model_inator import NoteStorage
from notes.note_util_inator import _load_note, _note_exists
from notes.sync_service_inator import sync_push_note

router_sync = APIRouter(prefix="/sync", tags=["sync"])


@router_sync.post("/push/{bubble_id}/{note_id}")
def push(
    bubble_id: str,
    note_id: str,
    incoming: NoteStorage,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    _assert_bubble_owner(bubble_id, user_id)
    result = sync_push_note(bubble_id, note_id, incoming)
    return {
        "note": result.note,
        "conflicted_pages": result.conflicted_pages,
        "server_vector": result.note.metadata.version_vector,
    }


@router_sync.get("/pull/{bubble_id}/{note_id}")
def pull(
    bubble_id: str,
    note_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    filename = f"{note_id}.json"
    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")
    note = _load_note(notes_dir, filename)
    return {"note": note, "server_vector": note.metadata.version_vector}


@router_sync.get("/replica-id")
def replica_id(user_id: str = Depends(get_current_user_id)):
    """The hub's stable replica id — the peer id a client keys its cursor on."""
    return {"replica_id": SyncStoreInator().get_or_create_replica_id("hub")}
