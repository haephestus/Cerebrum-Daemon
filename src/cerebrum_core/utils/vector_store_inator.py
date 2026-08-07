"""
cerebrum_core.utils.vector_store_inator
========================================
The single concrete vector store for the whole codebase.

Everything vector-shaped in Cerebrum is a folder of FAISS files (see
faiss_store_inator) — the KB archives, the per-note retrieval cache, the
engram/answer cache. They were each hand-rolling load/add/filter/save on
top of langchain FAISS. FaissVectorStore wraps that folder model behind one
small, uniform surface (add / search / get / delete / count / clear) so:

  * callers stop repeating the get_or_create_store -> add -> save_store dance,
  * metadata filtering is done one consistent way (client-side, since FAISS's
    own `filter=` support is version-dependent and unreliable),
  * the flexible cache layer (cache_inator) and the grading retriever can be
    built *around* this instead of each owning a FAISS store directly.

A store IS its folder: `persist_dir` fully identifies it, exactly like
faiss_store_inator/delete_store and the KnowledgebaseManager domain/subject
paths. Nothing here changes where anything is stored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from cerebrum_core.utils.embeddings_inator import get_embeddings
from cerebrum_core.utils.faiss_store_inator import (
    delete_by_metadata,
    delete_store,
    get_by_metadata,
    get_or_create_store,
    iter_docs,
    matches_filter,
    save_store,
)

# A thing that can be added: either a ready Document, or a (text, metadata) pair.
Addable = Union[Document, tuple[str, Optional[dict]]]


class FaissVectorStore:
    """A single FAISS store rooted at `persist_dir`.

    Cheap to construct (no I/O until a method touches the store); each
    operation loads the on-disk index through get_or_create_store, so an
    instance is safe to keep around or throw away per call.
    """

    def __init__(
        self,
        persist_dir: Union[str, Path],
        embeddings: Optional[OllamaEmbeddings] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        # One shared embeddings client by default (see embeddings_inator).
        self.embeddings = embeddings or get_embeddings(embedding_model)

    # -- internal ---------------------------------------------------------
    def _store(self) -> FAISS:
        return get_or_create_store(self.persist_dir, self.embeddings)

    @staticmethod
    def _to_documents(items: Iterable[Addable]) -> list[Document]:
        docs: list[Document] = []
        for item in items:
            if isinstance(item, Document):
                docs.append(item)
            else:
                text, metadata = item
                docs.append(Document(page_content=text, metadata=metadata or {}))
        return docs

    # -- writes -----------------------------------------------------------
    def add(self, items: Iterable[Addable], *, save: bool = True) -> list[str]:
        """Add Documents (or (text, metadata) pairs). Persists by default —
        pass save=False to batch several adds and save_store() once."""
        docs = self._to_documents(items)
        if not docs:
            return []
        store = self._store()
        ids = store.add_documents(docs)
        if save:
            save_store(store, self.persist_dir)
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        *,
        save: bool = True,
    ) -> list[str]:
        metas = list(metadatas) if metadatas is not None else None
        pairs: list[Addable] = [
            (t, (metas[i] if metas is not None and i < len(metas) else {}))
            for i, t in enumerate(texts)
        ]
        return self.add(pairs, save=save)

    # -- reads ------------------------------------------------------------
    def search(
        self, query: str, k: int = 5, filter: Optional[dict[str, Any]] = None
    ) -> list[Document]:
        """Semantic top-k. When `filter` is given, over-fetch and match
        metadata client-side (FAISS's native filter support is unreliable)."""
        store = self._store()
        fetch_k = k * 4 if filter else k
        candidates = store.similarity_search(query, k=fetch_k)
        if filter:
            candidates = [
                d for d in candidates if matches_filter(d.metadata or {}, filter)
            ]
        return candidates[:k]

    def get(self, filter: dict[str, Any]) -> list[Document]:
        """Deterministic metadata lookup (linear scan + equality match)."""
        return [doc for _, doc in get_by_metadata(self._store(), filter)]

    def all(self) -> list[Document]:
        return [doc for _, doc in iter_docs(self._store())]

    # -- lifecycle --------------------------------------------------------
    def delete(self, filter: dict[str, Any]) -> int:
        """Delete every document matching `filter`; returns the count deleted."""
        return delete_by_metadata(self._store(), self.persist_dir, filter)

    def exists(self) -> bool:
        return (self.persist_dir / "index.faiss").exists()

    def count(self) -> int:
        return len(self._store().index_to_docstore_id)

    def clear(self) -> None:
        """Drop the whole store (the folder). Mirrors Chroma's delete_collection."""
        delete_store(self.persist_dir)
