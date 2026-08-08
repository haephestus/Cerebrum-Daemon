"""
sources.registry — the set of available external suggestion providers.

All keyless + free. Content licenses vary per result and are resolved by
common.license_policy_inator downstream (ingest vs pointer). The orchestrator
skips network providers when offline; OpenStax (local catalog) always runs.
"""

from __future__ import annotations

from sources import SearchProvider
from sources.doab_inator import DOABSource
from sources.gutenberg_inator import GutenbergSource
from sources.mediawiki_inator import WikibooksSource, WikipediaSource
from sources.ncbi_bookshelf_inator import NCBIBookshelfSource
from sources.openalex_inator import OpenAlexSource
from sources.openstax_inator import OpenStaxSource
from sources.semantic_scholar_inator import SemanticScholarSource

EXTERNAL_PROVIDERS: list[SearchProvider] = [
    OpenAlexSource(),
    SemanticScholarSource(),
    WikipediaSource(),
    WikibooksSource(),
    GutenbergSource(),
    DOABSource(),
    NCBIBookshelfSource(),
    OpenStaxSource(),
]


def external_providers(enabled: set[str] | None = None) -> list[SearchProvider]:
    """The external providers, optionally filtered to a set of names."""
    if enabled is None:
        return list(EXTERNAL_PROVIDERS)
    return [p for p in EXTERNAL_PROVIDERS if p.name in enabled]
