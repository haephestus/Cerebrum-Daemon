"""OpenAlex — scholarly works. Metadata is CC0; each work's *content* license
varies (read from the OA location), so unknown → pointer downstream."""

from __future__ import annotations

from sources import Seed, SuggestedReading
from sources._http import get_json


def _abstract(work: dict) -> str:
    idx = work.get("abstract_inverted_index")
    if not idx:
        return ""
    positions = [(loc, word) for word, locs in idx.items() for loc in locs]
    positions.sort()
    return " ".join(w for _, w in positions)


class OpenAlexSource:
    name = "openalex"
    requires_network = True

    def search(self, seed: Seed, k: int = 5) -> list[SuggestedReading]:
        q = seed.primary_query()
        if not q:
            return []
        data = get_json(
            "https://api.openalex.org/works",
            params={"search": q, "per-page": k, "sort": "relevance_score:desc"},
        )
        if not data:
            return []
        out: list[SuggestedReading] = []
        for i, w in enumerate(data.get("results", [])):
            loc = w.get("best_oa_location") or w.get("primary_location") or {}
            url = loc.get("landing_page_url") or loc.get("pdf_url") or w.get("id")
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:3]
            )
            snippet = _abstract(w)
            out.append(
                SuggestedReading(
                    title=w.get("title") or w.get("display_name") or "Untitled",
                    source=self.name,
                    url=url,
                    license=loc.get("license"),
                    snippet=((f"{authors}. " if authors else "") + snippet)[:280],
                    reason="Scholarly work relevant to your topic.",
                    addresses=list(seed.weak_areas),
                    score=round(1.0 / (1 + i), 4),
                )
            )
        return out
