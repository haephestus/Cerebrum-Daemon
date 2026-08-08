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

import logging

from sources import Seed, SuggestedReading
from sources.kb_source_inator import KnowledgeBaseSource

logger = logging.getLogger(__name__)

# Sources that are books/textbooks vs. short reference articles vs. papers —
# used to bias ranking toward a learner's effective profile.
_BOOK_SOURCES = {"openstax", "gutenberg", "doab", "ncbi_bookshelf"}
_PAPER_SOURCES = {"openalex", "semantic_scholar"}
_ARTICLE_SOURCES = {"wikipedia", "wikibooks"}


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


def _profile_bias(source: str, effective: dict | None) -> float:
    """Small ±0.1 nudge toward source types that suit the learner's effective
    profile: depth-first → books; breadth-first → articles; abstract → papers;
    concrete → textbooks. No profile → no bias."""
    if not effective:
        return 0.0
    axes = effective.get("axes") or {}

    def v(name: str) -> float:
        return (axes.get(name) or {}).get("effective", 0.0)

    bias = 0.0
    depth = v("breadth_depth")          # + depth, - breadth
    if source in _BOOK_SOURCES:
        bias += 0.1 * max(0.0, depth)
    if source in _ARTICLE_SOURCES:
        bias += 0.1 * max(0.0, -depth)
    abstraction = v("abstraction_concrete")  # + abstract, - concrete
    if source in _PAPER_SOURCES:
        bias += 0.1 * max(0.0, abstraction)
    if source in _BOOK_SOURCES:
        bias += 0.08 * max(0.0, -abstraction)  # concrete → textbooks
    return round(bias, 4)


def _rank(readings: list[SuggestedReading], effective: dict | None) -> list[SuggestedReading]:
    """Re-score: KB first ("from what the user has"), ingestable-license nudge,
    profile bias, then the provider's own relevance."""
    from common import license_policy_inator as lp

    for r in readings:
        final = r.score
        if r.in_kb:
            final += 0.5                                   # user's own material first
        elif lp.is_ingestable(r.license):
            final += 0.15                                  # can graduate into the KB
        final += _profile_bias(r.source, effective)
        r.score = round(final, 4)
    readings.sort(key=lambda r: r.score, reverse=True)
    return readings


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


def suggest(
    repo,
    *,
    seed: Seed,
    seed_ref: str,
    user_id: str,
    manager,
    file_registry,
    org_ids: list,
    include_external: bool = False,
    enabled_providers: set | None = None,
    effective_profile: dict | None = None,
    k: int = 3,
    limit: int = 15,
    persist: bool = True,
) -> list[dict]:
    """Full suggestion run: Tier-1 KB always, plus external providers when
    `include_external` (each degrades to [] offline/on error). Merges, ranks
    (KB-first + ingestable nudge + learning-profile bias), persists candidates.

    KB-only (the default) is offline-safe and cheap; external is opt-in because
    it costs network + latency.
    """
    readings: list[SuggestedReading] = KnowledgeBaseSource(
        manager, file_registry, user_id, org_ids
    ).search(seed, k=k)

    if include_external:
        from sources.registry import external_providers

        for provider in external_providers(enabled_providers):
            try:
                readings.extend(provider.search(seed, k=k))
            except Exception as e:  # a broken provider must not sink the run
                logger.warning("provider %s failed: %s", getattr(provider, "name", "?"), e)

    ranked = _rank(readings, effective_profile)[:limit]
    rows = [_to_row(r) for r in ranked]
    if not persist:
        return rows
    repo.replace_candidate_suggestions(user_id, seed_ref, rows)
    return repo.list_suggestions(user_id, seed_ref)
