"""
api.routes_suggested_reading — suggested-reading endpoints (gap 3).

Phase 0: KB-first, offline. `GET /suggested-reading/note/{bubble_id}/{note_id}`
seeds from the note's cached analysis and returns readings already in the
knowledge base; `GET /suggested-reading/list` reads back persisted suggestions.
Accept/dismiss + external providers arrive in later phases.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from cerebrum_core.knowledgebase_inator import KnowledgebaseManager
from cerebrum_core.suggested_reading_inator import build_seed_from_overview, suggest
from cerebrum_core.user_context_inator import (
    build_effective_profile,
    get_current_user_id,
)
from common import license_policy_inator as license_policy
from common.cache_inator import AnalysisCacheInator
from common.file_util_inator import CerebrumPaths
from notes.note_util_inator import _load_note

logger = logging.getLogger(__name__)

router_suggested_reading = APIRouter(
    prefix="/suggested-reading", tags=["suggested-reading"]
)


def _load_overview(bubble_id: str, note_id: str):
    """(note_overview, resolved_note_id) for the note's current cached analysis,
    or (None, note_id) if the note or its analysis isn't available yet."""
    try:
        filename = note_id if note_id.endswith(".json") else f"{note_id}.json"
        notes_dir = CerebrumPaths().note_root_dir(bubble_id)
        note = _load_note(notes_dir, filename)
        overview = AnalysisCacheInator(
            bubble_id=bubble_id, note_id=note.note_id
        ).get_cached_overview(note.manifest.content_version)
        return overview, note.note_id
    except Exception as e:
        logger.info(
            "suggested-reading: no analysis overview for %s/%s: %s",
            bubble_id,
            note_id,
            e,
        )
        return None, note_id


@router_suggested_reading.get("/note/{bubble_id}/{note_id}")
def suggest_for_note(
    bubble_id: str,
    note_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    include_external: bool = Query(
        False, description="Also query free external providers (needs network)."
    ),
):
    """Suggested readings for a note, seeded from its analysis. KB-first
    (offline); pass include_external=true to also pull free OER/academic sources.
    Returns [] (with a hint) if the note hasn't been analysed yet."""
    repo = request.app.state.note_registry
    file_registry = request.app.state.file_registry

    overview, resolved_id = _load_overview(bubble_id, note_id)
    if not overview:
        return {
            "note_id": resolved_id,
            "suggestions": [],
            "message": "No analysis available yet — run analysis first.",
        }

    seed = build_seed_from_overview(overview)
    manager = KnowledgebaseManager()
    org_ids = repo.get_user_org_ids(user_id)
    effective = build_effective_profile(repo, user_id)
    suggestions = suggest(
        repo,
        seed=seed,
        seed_ref=resolved_id,
        user_id=user_id,
        manager=manager,
        file_registry=file_registry,
        org_ids=org_ids,
        include_external=include_external,
        effective_profile=effective,
    )
    return {
        "note_id": resolved_id,
        "seed": {
            "topic": seed.topic,
            "weak_areas": seed.weak_areas,
            "knowledge_gaps": seed.knowledge_gaps,
        },
        "suggestions": suggestions,
    }


@router_suggested_reading.get("/list")
def list_suggestions(
    request: Request,
    seed_ref: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    """Read back a user's persisted suggestions (optionally for one seed)."""
    repo = request.app.state.note_registry
    return {"suggestions": repo.list_suggestions(user_id, seed_ref)}


def _owned_suggestion(repo, user_id: str, suggestion_id: str) -> dict:
    sug = repo.get_suggestion(suggestion_id)
    if not sug or sug.get("user_id") != user_id:
        # 404 (not 403) so we don't reveal that someone else's id exists.
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return sug


@router_suggested_reading.post("/accept/{suggestion_id}")
def accept_suggestion(
    suggestion_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Accept a suggestion.

    - KB reading (already in the KB): just pin it as 'accepted'.
    - External reading: the license decides — 'ingest' (fetch→embed into the
      KB) vs 'pointer' (surface the link only). Phase 0/1 records the decision;
      the actual fetch+embed lands with the external providers (Phase 2), which
      is where there's real content and a live embed to run it through.
    """
    repo = request.app.state.note_registry
    sug = _owned_suggestion(repo, user_id, suggestion_id)

    if sug.get("in_kb"):
        repo.set_suggestion_status(suggestion_id, "accepted")
        return {"id": suggestion_id, "status": "accepted", "in_kb": True}

    action, reason = license_policy.decide(sug.get("license"))
    if action == "pointer":
        repo.set_suggestion_status(suggestion_id, "pointer")
        return {"id": suggestion_id, "status": "pointer", "reason": reason}

    # Ingestable per license → fetch the content and graduate it into the KB.
    # Guarded: if fetch or embed fails (e.g. no embedding service, unfetchable
    # URL, HTML with trafilatura missing), keep 'accepted' as recorded intent
    # rather than 500 — the user's accept still stands.
    from cerebrum_core.kb_ingest_inator import ingest_external_document
    from common.content_fetch_inator import fetch_as_markdown

    try:
        md = fetch_as_markdown(sug.get("url"))
        if not md:
            repo.set_suggestion_status(suggestion_id, "accepted")
            return {
                "id": suggestion_id,
                "status": "accepted",
                "reason": reason,
                "note": "accepted, but content could not be fetched yet",
            }
        fp = ingest_external_document(
            markdown_text=md,
            title=sug["title"],
            file_registry=request.app.state.file_registry,
            source=sug.get("source", "external"),
            url=sug.get("url"),
            license=sug.get("license"),
        )
        repo.set_suggestion_status(
            suggestion_id, "ingested", in_kb=True, file_fingerprint=fp
        )
        return {"id": suggestion_id, "status": "ingested", "file_fingerprint": fp}
    except Exception as e:
        logger.warning("ingest failed for %s: %s", suggestion_id, e)
        repo.set_suggestion_status(suggestion_id, "accepted")
        return {
            "id": suggestion_id,
            "status": "accepted",
            "reason": reason,
            "note": "accepted; ingestion into the KB failed (needs the embedding service)",
        }


@router_suggested_reading.post("/dismiss/{suggestion_id}")
def dismiss_suggestion(
    suggestion_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Hide a suggestion (won't be replaced on the next candidate refresh)."""
    repo = request.app.state.note_registry
    _owned_suggestion(repo, user_id, suggestion_id)
    repo.set_suggestion_status(suggestion_id, "dismissed")
    return {"id": suggestion_id, "status": "dismissed"}
