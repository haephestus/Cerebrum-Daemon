"""
cerebrum_core.suggested_reading_inator
======================================
Domain orchestrator for suggested reading (gap 3).

Phase 0: KB-first, offline. Builds a Seed from a note's analysis overview
(topic + weak areas + confused links + knowledge gaps), runs the Tier-1
knowledge-base source, and persists the candidates. External (online) tiers and
the accept→ingest→KB loop arrive in later phases; the flow here is deliberately
provider-agnostic so they slot in without changing callers.
"""

from __future__ import annotations

from sources import Seed, SuggestedReading
from sources.kb_source_inator import KnowledgeBaseSource


def build_seed_from_overview(overview: dict | None) -> Seed:
    """Distil a note-analysis `note_overview` into a search Seed. Confused
    links become 'A vs B' weak-area queries; knowledge gaps ride alongside."""
    overview = overview or {}
    cmap = overview.get("concept_map") or {}
    weak = list(cmap.get("weak_areas") or [])
    confused = [
        f'{c.get("concept_a")} vs {c.get("concept_b")}'
        for c in (cmap.get("confused_links") or [])
        if c.get("concept_a") and c.get("concept_b")
    ]
    gaps = list(overview.get("knowledge_gaps_summary") or [])
    return Seed(
        topic=overview.get("topic", "") or "",
        weak_areas=weak + confused,
        knowledge_gaps=gaps,
    )


def _to_row(r: SuggestedReading) -> dict:
    return {
        "title": r.title,
        "source": r.source,
        "url": r.url,
        "file_fingerprint": r.file_fingerprint,
        "license": r.license,
        "snippet": r.snippet,
        "reason": r.reason,
        "addresses": r.addresses,
        "score": r.score,
        "in_kb": r.in_kb,
    }


def suggest_from_kb(
    repo,
    *,
    seed: Seed,
    seed_ref: str,
    user_id: str,
    manager,
    file_registry,
    org_ids: list,
    k: int = 3,
    limit: int = 12,
    persist: bool = True,
) -> list[dict]:
    """Tier-1 suggestion run: KB search for `seed`, persist as candidates for
    `seed_ref`, return the stored rows (or the raw rows if persist=False).

    `repo` is a NoteEngramRepository (SuggestedReadingMixin); `manager` a
    KnowledgebaseManager; `file_registry` a FileRegisterInator.
    """
    src = KnowledgeBaseSource(manager, file_registry, user_id, org_ids)
    readings = src.search(seed, k=k)[:limit]
    rows = [_to_row(r) for r in readings]
    if not persist:
        return rows
    repo.replace_candidate_suggestions(user_id, seed_ref, rows)
    return repo.list_suggestions(user_id, seed_ref)
