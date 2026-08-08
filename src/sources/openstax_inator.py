"""OpenStax — CC-BY open textbooks. OpenStax has no search API, so this matches
the seed against a small local catalog of its books by keyword. requires_network
is False: it works offline (and the catalog is commercial-safe, CC-BY)."""

from __future__ import annotations

from sources import Seed, SuggestedReading

# (title, slug, keyword tags). Slug → https://openstax.org/details/books/<slug>.
_CATALOG: list[tuple[str, str, str]] = [
    ("Introductory Statistics", "introductory-statistics", "statistics probability distribution regression"),
    ("University Physics Volume 1", "university-physics-volume-1", "physics mechanics motion force energy thermodynamics"),
    ("College Physics 2e", "college-physics-2e", "physics motion force energy waves electricity"),
    ("Biology 2e", "biology-2e", "biology cell genetics evolution ecology molecular"),
    ("Concepts of Biology", "concepts-biology", "biology cell genetics evolution"),
    ("Microbiology", "microbiology", "microbiology bacteria virus immunology"),
    ("Anatomy and Physiology 2e", "anatomy-and-physiology-2e", "anatomy physiology body organ"),
    ("Chemistry 2e", "chemistry-2e", "chemistry atom molecule reaction bond acid"),
    ("Calculus Volume 1", "calculus-volume-1", "calculus derivative integral limit function"),
    ("Precalculus 2e", "precalculus-2e", "precalculus trigonometry function algebra"),
    ("College Algebra 2e", "college-algebra-2e", "algebra equation polynomial function"),
    ("Principles of Economics 3e", "principles-economics-3e", "economics supply demand market macroeconomics microeconomics"),
    ("Psychology 2e", "psychology-2e", "psychology behaviour cognition brain memory"),
    ("Introduction to Sociology 3e", "introduction-sociology-3e", "sociology society culture social"),
    ("Astronomy 2e", "astronomy-2e", "astronomy star planet galaxy universe"),
    ("Principles of Data Science", "principles-data-science", "data science statistics machine learning data structures"),
]


class OpenStaxSource:
    name = "openstax"
    requires_network = False  # local catalog match — usable offline

    def search(self, seed: Seed, k: int = 3) -> list[SuggestedReading]:
        text = (seed.topic + " " + " ".join(seed.weak_areas)).lower()
        if not text.strip():
            return []
        scored: list[tuple[int, str, str]] = []
        for title, slug, tags in _CATALOG:
            hits = sum(1 for w in set(tags.split()) if w in text)
            if hits:
                scored.append((hits, title, slug))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[SuggestedReading] = []
        for hits, title, slug in scored[:k]:
            out.append(
                SuggestedReading(
                    title=title,
                    source=self.name,
                    url=f"https://openstax.org/details/books/{slug}",
                    license="CC-BY",
                    snippet="Openly-licensed (CC-BY) textbook.",
                    reason="Open textbook covering your topic.",
                    addresses=list(seed.weak_areas),
                    score=round(min(1.0, hits / 3.0), 4),
                )
            )
        return out
