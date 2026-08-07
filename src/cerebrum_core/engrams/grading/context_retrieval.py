"""
cerebrum_core.engrams.grading.context_retrieval
===============================================
Concrete RAG-context retrieval for grading. This replaces the abstract
VectorStore.retrieve_note_chunks path (which had no implementation and
always returned []) with the intended two-tier strategy:

  1. CACHED RETRIEVAL first — the per-note FAISS cache that the chunk
     analyser populates during analysis (RetrievalCacheInator, keyed by
     note_id + bubble_id). This is the cheapest, most note-specific source
     and is what the engram generator's retrieval already primed.

  2. MAIN KNOWLEDGE BASE as a fallback — when the cache is INSUFFICIENT
     (fewer than top_k distinct chunks), the shortfall is topped up from
     the shared FAISS archives via KnowledgebaseManager.search_across_
     collections. "Insufficient" is deliberately a simple count threshold:
     an empty/thin cache (e.g. a note analysed before its retrieval cache
     was warmed) still gets real context instead of grading blind.

Both backends are synchronous (FAISS + OllamaEmbeddings), so the async
grading worker calls this via asyncio.to_thread — same pattern
ai_grading.call_grading_model uses for the blocking model calls.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from cerebrum_core.knowledgebase_inator import KnowledgebaseManager
from cerebrum_core.utils.cache_inator import RetrievalCacheInator

logger = logging.getLogger(__name__)


def retrieve_grading_context(
    *,
    query: str,
    bubble_id: Optional[str],
    note_id: Optional[str],
    topic: Optional[str] = None,
    top_k: int = 6,
) -> list[str]:
    """Return up to `top_k` distinct context chunks for grading `query`.

    Draws from the note's cached retrieval first and augments from the main
    knowledge base only when the cache doesn't supply enough. Never raises:
    a failure in either tier is logged and treated as "that source
    contributed nothing", so grading degrades to whatever context is
    available rather than erroring the whole job.
    """
    chunks: list[str] = []
    seen: set[str] = set()

    def _add(texts: Iterable[Optional[str]]) -> None:
        for raw in texts:
            text = (raw or "").strip()
            if text and text not in seen:
                seen.add(text)
                chunks.append(text)

    # ── Tier 1: cached retrieval (populated by the chunk analyser) ────────
    if note_id and bubble_id:
        try:
            cache = RetrievalCacheInator(note_id=note_id, bubble_id=bubble_id)
            cached_docs = cache.semantic_fetch(query, k=top_k)
            _add(d.page_content for d in cached_docs)
            logger.info(
                "grading context: %d chunk(s) from cache for note %s",
                len(chunks),
                note_id,
            )
        except Exception:
            logger.warning(
                "grading context: cached retrieval failed for note %s",
                note_id,
                exc_info=True,
            )
    else:
        logger.info(
            "grading context: no note_id/bubble_id — skipping cache, "
            "going straight to the knowledge base"
        )

    # ── Tier 2: main knowledge base, only if the cache was insufficient ───
    if len(chunks) < top_k:
        needed = top_k - len(chunks)
        try:
            kb = KnowledgebaseManager()
            kb_results = kb.search_across_collections(query, k=needed)
            before = len(chunks)
            _add(r.get("content") for r in kb_results)
            logger.info(
                "grading context: cache insufficient (%d/%d) — added %d "
                "chunk(s) from the knowledge base",
                before,
                top_k,
                len(chunks) - before,
            )
        except Exception:
            logger.warning(
                "grading context: knowledge base fallback failed",
                exc_info=True,
            )

    return chunks[:top_k]
