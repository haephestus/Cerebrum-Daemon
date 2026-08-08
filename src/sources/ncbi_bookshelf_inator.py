"""NCBI Bookshelf via E-utilities (esearch → esummary), db=books.

Most Bookshelf titles are free-to-read but publisher-copyrighted, so license is
left None → the policy surfaces them as POINTERS (not ingested) on a commercial
build. The OA/CC subset, when a license is known, can still be ingested."""

from __future__ import annotations

from sources import Seed, SuggestedReading
from sources._http import get_json

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class NCBIBookshelfSource:
    name = "ncbi_bookshelf"
    requires_network = True

    def search(self, seed: Seed, k: int = 5) -> list[SuggestedReading]:
        q = seed.primary_query()
        if not q:
            return []
        es = get_json(
            _ESEARCH,
            params={"db": "books", "term": q, "retmax": k, "retmode": "json"},
        )
        ids = ((es or {}).get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        summ = get_json(
            _ESUMMARY,
            params={"db": "books", "id": ",".join(ids), "retmode": "json"},
        )
        result = (summ or {}).get("result") or {}
        out: list[SuggestedReading] = []
        for i, uid in enumerate(ids):
            item = result.get(uid) or {}
            acc = item.get("bookaccession") or f"NBK{uid}"
            out.append(
                SuggestedReading(
                    title=item.get("title", "Untitled"),
                    source=self.name,
                    url=f"https://www.ncbi.nlm.nih.gov/books/{acc}/",
                    license=None,  # free-to-read but usually copyrighted → pointer
                    snippet=(item.get("publishername") or item.get("bookname") or "")[:280],
                    reason="Free-to-read reference (pointer unless open-access).",
                    addresses=list(seed.weak_areas),
                    score=round(1.0 / (1 + i), 4),
                )
            )
        return out
