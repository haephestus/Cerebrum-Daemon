"""Project Gutenberg via the gutendex API (public domain — always ingestable)."""

from __future__ import annotations

from sources import Seed, SuggestedReading
from sources._http import get_json


class GutenbergSource:
    name = "gutenberg"
    requires_network = True

    def search(self, seed: Seed, k: int = 5) -> list[SuggestedReading]:
        q = seed.primary_query()
        if not q:
            return []
        data = get_json("https://gutendex.com/books", params={"search": q}, timeout=15.0)
        if not data:
            return []
        out: list[SuggestedReading] = []
        for i, b in enumerate((data.get("results") or [])[:k]):
            authors = ", ".join(a.get("name", "") for a in (b.get("authors") or [])[:2])
            fmts = b.get("formats") or {}
            url = (
                fmts.get("text/html")
                or fmts.get("text/plain; charset=utf-8")
                or fmts.get("application/epub+zip")
                or next(iter(fmts.values()), None)
            )
            out.append(
                SuggestedReading(
                    title=b.get("title", "Untitled"),
                    source=self.name,
                    url=url,
                    license="public-domain",
                    snippet=authors,
                    reason="Public-domain book relevant to your topic.",
                    addresses=list(seed.weak_areas),
                    score=round(1.0 / (1 + i), 4),
                )
            )
        return out
