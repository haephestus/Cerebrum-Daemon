"""
cerebrum_core.engrams.storage.vector_store
=============================================
Abstract vector store interface + helper functions for indexing engrams,
long-question answers, and retrieving RAG context for grading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from vectorstore import retrieve_inator

# ---------------------------------------------------------------------------
# Interfaces
#
# TODO: no concrete implementation of EmbeddingProvider or VectorStore exists
# in this codebase — everything below (index_note, index_answer,
# retrieve_note_chunks, retrieve_past_answers, retrieve_similar_answers) is
# correct logic against these interfaces but has nothing to actually run
# against. worker.py's process_grading_job depends on both being real.
#
# TODO: engram_generator_inator.py already has its own embedding path via
# RetrieverInator (embedding_model comes from ConfigManager, same as the
# chat_model used for ollama_cloud_call). Check whether RetrieverInator (or
# whatever backs it — commented-out imports in that file suggest Chroma via
# FAISS should be wrapped here as the concrete
# EmbeddingProvider/VectorStore implementation, instead of standing up a
# second, separate embedding stack.
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self,
        namespace: str,
        records: list[dict[str, Any]],
    ) -> None: ...

    @abstractmethod
    async def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def delete(self, namespace: str, ids: list[str]) -> None: ...


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

NAMESPACE_NOTE_CHUNKS = "note_chunks"
NAMESPACE_ANSWERS = "answers"


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Index a note (splits into overlapping chunks)
#
# TODO: nothing in mastery_service.py, worker.py, or engram_generator_inator.py
# calls index_note — there's no wired-up "note created/updated -> chunk ->
# embed -> upsert" entrypoint yet. retrieve_note_chunks (used for RAG grading
# context) will always return [] until something calls this on note save.
# ---------------------------------------------------------------------------


async def index_note(
    store: VectorStore,
    embedder: EmbeddingProvider,
    note_id: str,
    content: str,
    topic: str,
    tags: list[str],
    subtopic: Optional[str] = None,
) -> list[str]:
    chunks = chunk_text(content)
    vectors = await embedder.embed_batch(chunks)

    records = [
        {
            "id": f"note:{note_id}:chunk:{i}",
            "vector": vectors[i],
            "metadata": {
                "note_id": note_id,
                "topic": topic,
                "subtopic": subtopic or "",
                "chunk_index": i,
                "text": chunk,
                "tags": ",".join(tags),
            },
        }
        for i, chunk in enumerate(chunks)
    ]
    await store.upsert(NAMESPACE_NOTE_CHUNKS, records)
    return [r["id"] for r in records]


# ---------------------------------------------------------------------------
# Index a long-question answer
# ---------------------------------------------------------------------------


async def index_answer(
    store: VectorStore,
    embedder: EmbeddingProvider,
    *,
    attempt_id: str,
    engram_id: str,
    user_id: str,
    topic: str,
    answer: str,
    score: float,
    target_cognitive_level: int,
    attempted_at: str,
) -> str:
    vector = await embedder.embed(answer)
    await store.upsert(
        NAMESPACE_ANSWERS,
        [
            {
                "id": attempt_id,
                "vector": vector,
                "metadata": {
                    "attempt_id": attempt_id,
                    "engram_id": engram_id,
                    "user_id": user_id,
                    "topic": topic,
                    "score": score,
                    "target_cognitive_level": target_cognitive_level,
                    "attempted_at": attempted_at,
                    "text": answer[:1000],  # truncated for metadata
                },
            }
        ],
    )
    return attempt_id


# ---------------------------------------------------------------------------
# Retrieve note chunks for RAG (grading context)
# ---------------------------------------------------------------------------


async def retrieve_note_chunks(
    store: VectorStore,
    embedder: EmbeddingProvider,
    query: str,
    topic: Optional[str] = None,
    note_id: Optional[str] = None,
    top_k: int = 5,
) -> list[str]:
    vector = await embedder.embed(query)
    meta_filter: dict[str, Any] = {}
    if topic:
        meta_filter["topic"] = topic
    if note_id:
        meta_filter["note_id"] = note_id

    results = await store.query(
        NAMESPACE_NOTE_CHUNKS,
        vector,
        top_k,
        meta_filter or None,
    )
    return [r["metadata"]["text"] for r in results if r.get("metadata", {}).get("text")]


# ---------------------------------------------------------------------------
# Retrieve user's past answers for same engram (regression context)
# ---------------------------------------------------------------------------


@dataclass
class PastAnswer:
    text: str
    score: float
    attempted_at: str


async def retrieve_past_answers(
    store: VectorStore,
    embedder: EmbeddingProvider,
    *,
    current_answer: str,
    engram_id: str,
    user_id: str,
    limit: int = 3,
) -> list[PastAnswer]:
    vector = await embedder.embed(current_answer)
    results = await store.query(
        NAMESPACE_ANSWERS,
        vector,
        limit,
        {"engram_id": engram_id, "user_id": user_id},
    )
    return [
        PastAnswer(
            text=r["metadata"].get("text", ""),
            score=float(r["metadata"].get("score", 0)),
            attempted_at=r["metadata"].get("attempted_at", ""),
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# Retrieve semantically similar answers across all users (calibration)
# ---------------------------------------------------------------------------


async def retrieve_similar_answers(
    store: VectorStore,
    embedder: EmbeddingProvider,
    answer: str,
    engram_id: str,
    exclude_user_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    vector = await embedder.embed(answer)
    results = await store.query(
        NAMESPACE_ANSWERS, vector, top_k + 5, {"engram_id": engram_id}
    )
    return [
        {
            "text": r["metadata"].get("text", ""),
            "score": r["metadata"].get("score", 0),
            "target_cognitive_level": r["metadata"].get("target_cognitive_level", 1),
        }
        for r in results
        if r["metadata"].get("user_id") != exclude_user_id
    ][:top_k]
