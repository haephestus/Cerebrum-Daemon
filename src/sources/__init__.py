"""
sources — content-source adapters for suggested reading.

Each adapter implements SearchProvider and returns SuggestedReading candidates.
Top-level infra package (sibling of api/, cerebrum_core/); must NOT import from
cerebrum_core. Tier-1 is the internal KB source (offline, this phase); Tier-2
academic/OER adapters (online-only) arrive in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Seed:
    """What to look readings up against, distilled from a note's analysis
    (+ study plan / learning profile in later phases)."""

    topic: str = ""
    weak_areas: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)

    def queries(self) -> list[tuple[str, str]]:
        """(query_text, addresses) pairs to search on. Each weak area / gap is
        its own query so a hit can be attributed to *what* it addresses; the
        bare topic is a catch-all when nothing more specific exists."""
        pairs: list[tuple[str, str]] = []
        for wa in self.weak_areas:
            pairs.append((f"{self.topic} {wa}".strip(), wa))
        for g in self.knowledge_gaps:
            pairs.append((f"{self.topic} {g}".strip(), g))
        if not pairs and self.topic:
            pairs.append((self.topic, self.topic))
        return pairs

    def primary_query(self, max_areas: int = 2) -> str:
        """One compact query string for external providers (which we don't want
        to hit once per weak area): topic + the first couple of weak areas."""
        parts = [self.topic] + self.weak_areas[:max_areas]
        return " ".join(p for p in parts if p).strip()


@dataclass
class SuggestedReading:
    """One reading suggestion. Phase-0 fills the KB fields; external providers
    (later) fill url/license and leave file_fingerprint None until ingested."""

    title: str
    source: str                       # 'knowledgebase' now; provider name later
    snippet: str = ""
    url: Optional[str] = None
    file_fingerprint: Optional[str] = None
    license: Optional[str] = None
    reason: str = ""
    addresses: list[str] = field(default_factory=list)
    score: float = 0.0
    in_kb: bool = False


@runtime_checkable
class SearchProvider(Protocol):
    """A source of reading suggestions. `requires_network` lets the orchestrator
    skip online providers when offline (KB stays available)."""

    name: str
    requires_network: bool

    def search(self, seed: Seed, k: int) -> list[SuggestedReading]: ...
