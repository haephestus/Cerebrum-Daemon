"""
sources.kb_source_inator — Tier-1 suggested-reading provider over the local
knowledge base.

Offline: similarity search across the user's *visible* KB collections, grouped
to source documents. This is the primary tier ("first from what the user has").
Access is enforced with the same `file_registry.filter_visible` used by the KB
search route, so private files never leak into suggestions.
"""

from __future__ import annotations

import logging
from typing import Any

from sources import Seed, SuggestedReading

logger = logging.getLogger(__name__)


class KnowledgeBaseSource:
    name = "knowledgebase"
    requires_network = False

    def __init__(self, manager, file_registry, user_id: str, org_ids: list):
        self._manager = manager
        self._files = file_registry
        self._user_id = user_id
        self._org_ids = org_ids or []

    def search(self, seed: Seed, k: int = 3) -> list[SuggestedReading]:
        # Accumulate at the DOCUMENT level (keyed by file_fingerprint): many
        # chunk hits across several weak-area queries collapse to one reading
        # per source document, carrying the best score + every area it addresses.
        by_doc: dict[str, dict] = {}
        for query, addresses in seed.queries():
            if not query.strip():
                continue
            try:
                hits = self._manager.search_across_collections(query=query, k=k)
            except Exception as e:
                logger.warning("KB suggested-reading search failed for %r: %s", query, e)
                continue
            for rank, hit in enumerate(hits):
                meta = hit.get("metadata") or {}
                fp = meta.get("file_fingerprint")
                if not fp:
                    continue  # can't identify/access-check the doc — skip
                score = 1.0 / (1 + rank)  # similarity_search returns best-first
                slot = by_doc.setdefault(
                    fp, {"score": 0.0, "addresses": set(), "snippet": ""}
                )
                slot["score"] = max(slot["score"], score)
                if addresses:
                    slot["addresses"].add(addresses)
                if not slot["snippet"]:
                    slot["snippet"] = (hit.get("content") or "")[:280]

        if not by_doc:
            return []

        # Access filter: only files this user (or one of their orgs) may see.
        visible = set(
            self._files.filter_visible(list(by_doc), self._user_id, self._org_ids)
        )

        readings: list[SuggestedReading] = []
        for fp, slot in by_doc.items():
            if fp not in visible:
                continue
            info = self._doc_info(fp)
            title = info.get("original_name") or info.get("sanitized_name") or fp
            readings.append(
                SuggestedReading(
                    title=title,
                    source=self.name,
                    snippet=slot["snippet"],
                    file_fingerprint=fp,
                    in_kb=True,
                    addresses=sorted(a for a in slot["addresses"] if a),
                    reason="Already in your knowledge base and relevant to your weak areas.",
                    score=round(slot["score"], 4),
                )
            )
        readings.sort(key=lambda r: r.score, reverse=True)
        return readings

    def _doc_info(self, fp: str) -> dict[str, Any]:
        try:
            return self._files.get_by_fingerprint(fp) or {}
        except Exception:
            return {}
