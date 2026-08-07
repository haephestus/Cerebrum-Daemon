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

from fastapi import APIRouter, Depends, Request

from cerebrum_core.knowledgebase_inator import KnowledgebaseManager
from cerebrum_core.suggested_reading_inator import (
    build_seed_from_overview,
    suggest_from_kb,
)
from cerebrum_core.user_context_inator import get_current_user_id
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
        ).get_cached_overview(note.metadata.content_version)
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
):
    """KB-first suggested readings for a note, seeded from its analysis.
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
    suggestions = suggest_from_kb(
        repo,
        seed=seed,
        seed_ref=resolved_id,
        user_id=user_id,
        manager=manager,
        file_registry=file_registry,
        org_ids=org_ids,
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
