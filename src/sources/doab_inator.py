"""Directory of Open Access Books (DOAB) via its DSpace discovery API.

Open-access academic books; license VARIES per book (read from dc.rights /
metadata) so the policy filter decides ingest vs pointer per result. Parses the
DSpace search response defensively — an unexpected shape just yields []."""

from __future__ import annotations

from sources import Seed, SuggestedReading
from sources._http import get_json

_API = "https://directory.doabooks.org/server/api/discover/search/objects"


def _meta(md: dict, key: str) -> str:
    vals = md.get(key) or []
    if vals and isinstance(vals, list):
        return vals[0].get("value", "") if isinstance(vals[0], dict) else str(vals[0])
    return ""


class DOABSource:
    name = "doab"
    requires_network = True

    def search(self, seed: Seed, k: int = 5) -> list[SuggestedReading]:
        q = seed.primary_query()
        if not q:
            return []
        data = get_json(_API, params={"query": q, "size": k})
        if not data:
            return []
        try:
            objects = (
                data.get("_embedded", {})
                .get("searchResult", {})
                .get("_embedded", {})
                .get("objects", [])
            )
        except Exception:
            return []
        out: list[SuggestedReading] = []
        for i, obj in enumerate(objects[:k]):
            io = (obj.get("_embedded") or {}).get("indexableObject") or {}
            md = io.get("metadata") or {}
            title = _meta(md, "dc.title") or io.get("name") or "Untitled"
            license_ = (
                _meta(md, "oapen.relation.isPublishedBy")  # rarely a license; fallbacks below
                or _meta(md, "dc.rights.uri")
                or _meta(md, "dc.rights")
                or None
            )
            handle = io.get("handle") or ""
            url = (
                _meta(md, "oapen.identifier.doi")
                or (f"https://directory.doabooks.org/handle/{handle}" if handle else None)
            )
            authors = _meta(md, "dc.contributor.author")
            out.append(
                SuggestedReading(
                    title=title,
                    source=self.name,
                    url=url,
                    license=license_,
                    snippet=(authors or _meta(md, "dc.description.abstract"))[:280],
                    reason="Open-access academic book relevant to your topic.",
                    addresses=list(seed.weak_areas),
                    score=round(1.0 / (1 + i), 4),
                )
            )
        return out
