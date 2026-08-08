"""Semantic Scholar Academic Graph (keyless). Metadata + abstract; full text
only when open-access (openAccessPdf) — otherwise a pointer downstream."""

from __future__ import annotations

from sources import Seed, SuggestedReading
from sources._http import get_json


class SemanticScholarSource:
    name = "semantic_scholar"
    requires_network = True

    def search(self, seed: Seed, k: int = 5) -> list[SuggestedReading]:
        q = seed.primary_query()
        if not q:
            return []
        data = get_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": q,
                "limit": k,
                "fields": "title,abstract,url,openAccessPdf,authors,year",
            },
        )
        if not data:
            return []
        out: list[SuggestedReading] = []
        for i, p in enumerate(data.get("data") or []):
            oa = p.get("openAccessPdf") or {}
            authors = ", ".join(
                a.get("name", "") for a in (p.get("authors") or [])[:3]
            )
            out.append(
                SuggestedReading(
                    title=p.get("title", "Untitled"),
                    source=self.name,
                    url=oa.get("url") or p.get("url"),
                    license=oa.get("license"),
                    snippet=(
                        (f"{authors}. " if authors else "") + (p.get("abstract") or "")
                    )[:280],
                    reason="Academic paper relevant to your topic.",
                    addresses=list(seed.weak_areas),
                    score=round(1.0 / (1 + i), 4),
                )
            )
        return out
