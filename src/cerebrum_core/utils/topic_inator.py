"""
cerebrum_core.utils.topic_inator
=================================
`topic` is the stringly-typed key that engrams, mastery, and study-plan
weeks all group and cross-DB-join on (there's no topic entity/id). That
makes it fragile: two LLM outputs like "CRISPR Gene Drive " and
"CRISPR  Gene Drive" would fragment a student's mastery into two buckets.

normalize_topic() is the conservative, non-breaking first line of defence:
it canonicalises whitespace and unicode form at the write boundary so
trivial variants collapse to one key, WITHOUT lowercasing (topic is shown
to users, so case is preserved). A stronger, case-insensitive canonical
key — or a real topic table with ids — is the follow-up if fragmentation
persists; this deliberately doesn't change stored casing/content.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_WS = re.compile(r"\s+")


def normalize_topic(topic: Optional[str]) -> Optional[str]:
    """Trim, collapse internal whitespace, and NFKC-normalise a topic.
    Returns None/empty unchanged so callers can pass optional topics through.
    This is the canonical *display* form — case is preserved."""
    if not topic:
        return topic
    normalised = unicodedata.normalize("NFKC", topic).strip()
    return _WS.sub(" ", normalised)


def topic_slug(topic: Optional[str]) -> Optional[str]:
    """The canonical *identity* key for a topic: normalize_topic() plus
    casefold, so "CRISPR Gene Drive", "crispr gene drive", and
    "CRISPR  Gene  Drive " all map to one slug (and therefore one topic
    entity). Returns None/empty unchanged. The topics table is UNIQUE on
    (user_id, slug); the human-facing name keeps its first-seen casing."""
    normalised = normalize_topic(topic)
    return normalised.casefold() if normalised else normalised
