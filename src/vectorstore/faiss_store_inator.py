"""
Shared helpers for FAISS-backed vector stores. Anything that used to
instantiate `Chroma(...)` directly should go through these instead of
hand-rolling its own load/save/filter/delete logic.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def get_or_create_store(persist_dir: Path, embeddings: OllamaEmbeddings) -> FAISS:
    persist_dir.mkdir(parents=True, exist_ok=True)

    if (persist_dir / "index.faiss").exists():
        return FAISS.load_local(
            folder_path=str(persist_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    dim = len(embeddings.embed_query("dimension probe"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def save_store(store: FAISS, persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(persist_dir))


def delete_store(persist_dir: Path) -> None:
    """Equivalent of Chroma's delete_collection() — the collection IS the folder."""
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


def iter_docs(store: FAISS):
    """Yield (doc_id, Document) for every document currently in the store."""
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        if isinstance(doc, Document):
            yield doc_id, doc


def matches_filter(metadata: dict, metadata_filter: Dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in metadata_filter.items())


def get_by_metadata(
    store: FAISS, metadata_filter: Dict[str, Any]
) -> List[Tuple[str, Document]]:
    """Equivalent of Chroma's `.get(where=...)` — linear scan + equality match."""
    return [
        (doc_id, doc)
        for doc_id, doc in iter_docs(store)
        if matches_filter(doc.metadata or {}, metadata_filter)
    ]


def delete_by_metadata(
    store: FAISS, persist_dir: Path, metadata_filter: Dict[str, Any]
) -> int:
    matches = get_by_metadata(store, metadata_filter)
    if not matches:
        return 0
    ids = [doc_id for doc_id, _ in matches]
    store.delete(ids)
    save_store(store, persist_dir)
    return len(ids)
