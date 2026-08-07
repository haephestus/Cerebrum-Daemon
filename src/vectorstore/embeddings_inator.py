"""
vectorstore.embeddings_inator
======================================
The single place OllamaEmbeddings is constructed.

Before this, `OllamaEmbeddings(model=...)` was instantiated ad hoc in ~6
modules (knowledgebase, cache, archive, retrieve, chunk_analyser), each
re-reading the embedding model from config in its own way. That made the
embedding backend impossible to swap or mock in one place and meant every
call site paid a fresh client construction.

get_embeddings() resolves the configured embedding model once, caches one
client per model name (the clients are stateless HTTP wrappers, so sharing
is safe), and is the only embedding entrypoint the rest of the codebase
should use.
"""

from __future__ import annotations

from typing import Optional

from langchain_ollama import OllamaEmbeddings

from cerebrum_core.user_inator import ConfigManager

# model name -> shared client. OllamaEmbeddings holds no per-call state, so
# one instance per model is reusable across threads/requests.
_CLIENTS: dict[str, OllamaEmbeddings] = {}


def resolve_embedding_model(model: Optional[str] = None) -> str:
    """Return the embedding model to use: the explicit argument if given,
    else the configured one. Raises if neither is available so callers fail
    loudly at construction rather than at first embed()."""
    if model:
        return model
    configured = ConfigManager().load_config().models.embedding_model
    if not configured:
        raise ValueError("Embedding model not configured")
    return configured


def get_embeddings(model: Optional[str] = None) -> OllamaEmbeddings:
    """Shared OllamaEmbeddings for `model` (or the configured default)."""
    resolved = resolve_embedding_model(model)
    client = _CLIENTS.get(resolved)
    if client is None:
        client = OllamaEmbeddings(model=resolved)
        _CLIENTS[resolved] = client
    return client
