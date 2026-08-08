"""Wikipedia / Wikibooks via the MediaWiki search API (both CC-BY-SA)."""

from __future__ import annotations

import re

from sources import Seed, SuggestedReading
from sources._http import get_json

_TAG = re.compile(r"<[^>]+>")


class _MediaWikiSource:
    api = ""
    base_url = ""
    name = ""
    requires_network = True

    def search(self, seed: Seed, k: int = 5) -> list[SuggestedReading]:
        q = seed.primary_query()
        if not q:
            return []
        data = get_json(
            self.api,
            params={
                "action": "query",
                "list": "search",
                "srsearch": q,
                "srlimit": k,
                "srprop": "snippet",
                "format": "json",
            },
        )
        if not data:
            return []
        out: list[SuggestedReading] = []
        for i, hit in enumerate((data.get("query") or {}).get("search", [])):
            title = hit.get("title", "")
            snippet = _TAG.sub("", hit.get("snippet", "") or "")
            out.append(
                SuggestedReading(
                    title=title,
                    source=self.name,
                    url=self.base_url + title.replace(" ", "_"),
                    license="CC-BY-SA",
                    snippet=snippet[:280],
                    reason="Reference article relevant to your topic.",
                    addresses=list(seed.weak_areas),
                    score=round(1.0 / (1 + i), 4),
                )
            )
        return out


class WikipediaSource(_MediaWikiSource):
    name = "wikipedia"
    api = "https://en.wikipedia.org/w/api.php"
    base_url = "https://en.wikipedia.org/wiki/"


class WikibooksSource(_MediaWikiSource):
    name = "wikibooks"
    api = "https://en.wikibooks.org/w/api.php"
    base_url = "https://en.wikibooks.org/wiki/"
