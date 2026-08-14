import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Optional

from langchain_core.documents import Document

from common.file_util_inator import CerebrumPaths
from vectorstore.embeddings_inator import get_embeddings
from vectorstore.vector_store_inator import FaissVectorStore

logger = logging.getLogger(__name__)

# ============================================================================
# FLEXIBLE CACHE TYPES
# ----------------------------------------------------------------------------
# There are three kinds of cache in Cerebrum, and they are genuinely
# different *backends*, not one thing duplicated:
#
#   * semantic caches (retrieval docs, engram/long-answer history) — these
#     are vector stores, so they now sit ON TOP OF the one FaissVectorStore
#     primitive (VectorCache below) instead of each re-opening FAISS by hand.
#   * a deterministic, version-keyed cache (chunk analysis) — plain JSON
#     files keyed by content_version (AnalysisCacheInator).
#   * a history cache (AnalysisHistoryCache) — SQLite.
#
# "Flexible cache types" means: pick the backend per kind via CacheType /
# make_cache, while every kind keeps EXACTLY the same on-disk location it
# used before (see each class's cache path) — nothing is migrated. The old
# class names (RetrievalCacheInator, EngramCacheInator, AnalysisCacheInator,
# AnalysisHistoryCache) are preserved as the public surface; the vector ones
# are now thin shims over VectorCache.
# ============================================================================


class CacheType(str, Enum):
    RETRIEVAL = "retrieval"  # semantic — cached RAG docs per note
    ENGRAM = "engram"  # semantic — historic long-answer docs per note
    ANALYSIS = "analysis"  # deterministic — version-keyed JSON chunk files (CURRENT)
    ANALYSIS_VECTOR = (
        "analysis_vector"  # semantic — completed analyses over time (HISTORY)
    )
    HISTORY = "history"  # sqlite — analysis history across versions


class VectorCache:
    """Semantic cache built *around* FaissVectorStore.

    Identity metadata (e.g. {note_id, bubble_id}) is stamped on every cached
    document and used to scope deterministic fetch / invalidation, so many
    notes can share one store folder yet stay separable. `scope_key` is the
    coarser key used to filter semantic search (historically bubble_id) so a
    query only ever surfaces docs from the right bubble.
    """

    def __init__(self, persist_dir, identity: dict, scope_key: str = "bubble_id"):
        self.persist_dir = persist_dir
        self.identity = identity
        self.scope_key = scope_key
        self.store = FaissVectorStore(persist_dir)

    def populate(self, docs: Optional[list[Document]]) -> None:
        if not docs:
            logger.warning("VectorCache.populate: nothing to cache")
            return
        stamped: list[Document] = []
        for d in docs:
            meta = dict(d.metadata) if d.metadata else {}
            meta.update(self.identity)
            meta.setdefault("cached_at", datetime.now().isoformat())
            stamped.append(Document(page_content=d.page_content, metadata=meta))
        try:
            self.store.add(stamped)
            logger.info(
                "VectorCache: cached %d doc(s) at %s", len(stamped), self.persist_dir
            )
        except Exception:
            logger.exception("VectorCache: failed to cache documents")

    def populate_texts(self, texts: list[str]) -> None:
        self.populate([Document(page_content=t, metadata={}) for t in texts if t])

    def fetch_all(self) -> Optional[list[Document]]:
        try:
            docs = self.store.get(self.identity)
        except Exception:
            logger.exception("VectorCache: deterministic fetch failed")
            return None
        return docs or None

    def semantic(self, query: str, k: int = 5) -> list[Document]:
        scope = (
            {self.scope_key: self.identity[self.scope_key]}
            if self.scope_key in self.identity
            else None
        )
        try:
            return self.store.search(query, k=k, filter=scope)
        except Exception:
            logger.exception("VectorCache: semantic search failed")
            return []

    def invalidate(self) -> int:
        try:
            return self.store.delete(self.identity)
        except Exception:
            logger.exception("VectorCache: invalidate failed")
            return 0

    def clear(self) -> None:
        try:
            self.store.clear()
        except Exception:
            logger.exception("VectorCache: clear failed")


# ============================================================================
# ANALYSIS CACHE - Use SQLite (fast, simple, version-based)
# ============================================================================


logger = logging.getLogger(__name__)


# TODO: analysis cache will now redirect from analysis dir to pages using write_page_analysis()
# in note_util_inator
class AnalysisCacheInator:
    """
    Simple file-based cache for note analysis.
    Cache key: note_id + content_version
    """

    def __init__(self, bubble_id: str, note_id: str):
        self.bubble_id = bubble_id
        self.note_id = note_id
        self.cache_note_dir = CerebrumPaths().note_analysis_dir(
            bubble_id=bubble_id, note_id=note_id
        )
        self.cache_note_dir.mkdir(parents=True, exist_ok=True)

    def get_cached_analysis(self, content_version: float) -> list[dict] | None:
        chunk_files = sorted(self.cache_note_dir.glob("chunk_*.json"))
        if not chunk_files:
            return None
        results = []
        for chunk_file in chunk_files:
            try:
                data = json.loads(chunk_file.read_text(encoding="utf-8"))
                if data.get("content_version") != content_version:
                    logger.info(f"Cache MISS — {chunk_file.name} is stale")
                    return None
                results.append(
                    {
                        "chunk_id": chunk_file.stem,  # "chunk_0", "chunk_1" etc from filename
                        "chunk_diagnostics": data["analysis"]["chunk_diagnostics"],
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupt cache file {chunk_file.name}: {e}")
                return None
        logger.info(f"Cache HIT for note {self.note_id} — {len(results)} chunks")
        return results

    def get_cache_info(self) -> Optional[dict]:
        """Get metadata about cached analysis from the first chunk."""
        chunk_files = sorted(self.cache_note_dir.glob("chunk_*.json"))
        if not chunk_files:
            return None
        try:
            cache_data = json.loads(chunk_files[0].read_text(encoding="utf-8"))
            return {
                "content_version": cache_data.get("content_version"),
                "cached_at": cache_data.get("cached_at"),
                "metadata": cache_data.get("metadata", {}),
            }
        except (json.JSONDecodeError, KeyError):
            return None

    # TODO: get note overview from manifest.json file, overview key.
    def get_cached_overview(self, content_version: float) -> Optional[dict]:
        """
        Return the note-level overview (topic, mastery signal, concept map,
        priority study areas, suggested sources) for a cached version.
        note_overview is aggregate-level — any chunk file carries the same
        copy — so reading the first one is sufficient.
        """
        chunk_files = sorted(self.cache_note_dir.glob("chunk_*.json"))
        if not chunk_files:
            return None
        try:
            data = json.loads(chunk_files[0].read_text(encoding="utf-8"))
            if data.get("content_version") != content_version:
                logger.info("Overview cache MISS — stale version")
                return None
            return data["analysis"].get("note_overview")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt cache file reading overview: {e}")
            return None

    def invalidate_cache(self) -> None:
        """Delete all cached chunk files for this note."""
        chunk_files = self.cache_note_dir.glob("chunk_*.json")
        deleted = 0
        for f in chunk_files:
            f.unlink()
            deleted += 1
        if deleted:
            logger.info(f"Invalidated {deleted} cache chunks for note {self.note_id}")

    def cache_analysis(
        self,
        content_version: float,
        analysis: dict,
        chunk_index: int,
        metadata: Optional[dict] = None,
    ) -> None:
        cache_data = {
            "note_id": self.note_id,
            "bubble_id": self.bubble_id,
            "content_version": content_version,
            "analysis": analysis,
            "cached_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        # Use cache_note_dir directly — self.cache_file is gone
        cache_file = self.cache_note_dir / f"chunk_{chunk_index}.json"
        cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        logger.info(
            f"Cached analysis chunk {chunk_index} for note {self.note_id} v{content_version}"
        )


# ============================================================================
# Engram CACHE - Use Chroma (semantic search makes sense here)
# ============================================================================


# TODO: implement engram caching
# caches only the historic answers of long questions
# question completion and progression live sql repo tracking


class EngramCacheInator:
    """ENGRAM cache type: historic long-answer text per note, semantic.

    Now a thin shim over VectorCache (which wraps the one FaissVectorStore).
    Same store folder as before — engram_archives_path(bubble_id) — and the
    same public methods, so callers are unchanged.
    """

    def __init__(self, note_id: str, bubble_id: str, engram_id=None) -> None:
        self.note_id = note_id
        self.bubble_id = bubble_id
        self.engram_id = engram_id
        self.cache_path = CerebrumPaths().engram_archives_path(bubble_id)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = VectorCache(
            self.cache_path, {"note_id": note_id, "bubble_id": bubble_id}
        )

    def cache_populator_inator(self, lq_response: str) -> None:
        if not lq_response:
            logger.warning(f"No documents to cache for note {self.note_id}")
            return
        self._cache.populate_texts([lq_response])

    def deterministic_fetcher(self) -> Optional[list[Document]]:
        return self._cache.fetch_all()

    def semantic_fetch(self, query: str, k: int = 5) -> list[Document]:
        return self._cache.semantic(query, k)

    def invalidate_note_cache(self) -> None:
        count = self._cache.invalidate()
        logger.info(f"Invalidated engram cache for note {self.note_id} ({count} docs)")

    def invalidate_bubble_cache(self) -> None:
        self._cache.clear()
        logger.info(f"Deleted entire engram cache for bubble {self.bubble_id}")


# ============================================================================
# RETRIEVAL CACHE - Use Chroma (semantic search makes sense here)
# ============================================================================


class RetrievalCacheInator:
    """RETRIEVAL cache type: RAG docs retrieved during analysis, semantic.

    Thin shim over VectorCache. Same store folder as before —
    note_analysis_dir(bubble_id, note_id) — and the same public methods, so
    the chunk analyser (its only caller) is unchanged.
    """

    def __init__(self, note_id: str, bubble_id: str) -> None:
        self.note_id = note_id
        self.bubble_id = bubble_id
        self.cache_path = CerebrumPaths().note_analysis_dir(
            bubble_id=bubble_id, note_id=note_id
        )
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = VectorCache(
            self.cache_path, {"note_id": note_id, "bubble_id": bubble_id}
        )

    def cache_populator_inator(self, retrieved_docs: Optional[list[Document]]) -> None:
        if not retrieved_docs:
            logger.warning(f"No documents to cache for note {self.note_id}")
            return
        self._cache.populate(retrieved_docs)

    def deterministic_fetcher(self) -> Optional[list[Document]]:
        return self._cache.fetch_all()

    def semantic_fetch(self, query: str, k: int = 5) -> list[Document]:
        return self._cache.semantic(query, k)

    def invalidate_note_cache(self) -> None:
        count = self._cache.invalidate()
        logger.info(
            f"Invalidated retrieval cache for note {self.note_id} ({count} docs)"
        )

    def invalidate_bubble_cache(self) -> None:
        self._cache.clear()
        logger.info(f"Deleted entire retrieval cache for bubble {self.bubble_id}")


class AnalysisVectorCacheInator:
    """ANALYSIS_VECTOR cache type: the HISTORICAL, queryable tier for analysis.

    AnalysisCacheInator holds the CURRENT analysis as version-keyed JSON
    (fast, exact). This persists each completed analysis — the note overview
    plus its findings — into a per-note vector store, tagged with
    content_version, so past analyses stay semantically queryable for overview
    development ("how has understanding of X evolved", "which findings mention
    Y"). It's the vectorstore analysis cache the completion path writes to
    once the JSON is up to date.

    Stored in an 'analysis_history' subfolder of the note's analysis dir so
    its index.faiss doesn't collide with the retrieval cache's index.faiss,
    which lives in that dir's root.
    """

    def __init__(self, note_id: str, bubble_id: str) -> None:
        self.note_id = note_id
        self.bubble_id = bubble_id
        self.cache_path = (
            CerebrumPaths().note_analysis_dir(bubble_id=bubble_id, note_id=note_id)
            / "analysis_history"
        )
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = VectorCache(
            self.cache_path, {"note_id": note_id, "bubble_id": bubble_id}
        )

    @staticmethod
    def _overview_text(overview: dict) -> str:
        """A readable, embeddable summary of an overview for semantic search."""
        parts = [
            str(overview.get("topic", "")),
            str(overview.get("mastery_signal", "")),
            str(overview.get("knowledge_gaps_summary", "")),
        ]
        for area in overview.get("priority_study_areas", []) or []:
            parts.append(str(area))
        cm = overview.get("concept_map", {}) or {}
        for key in ("strong_areas", "weak_areas"):
            for item in cm.get(key, []) or []:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)

    def persist(
        self, content_version, overview: dict, findings: Optional[list[dict]] = None
    ) -> None:
        """Append this analysis version to the historical vector store: one
        overview document plus one document per finding, all tagged with
        content_version so a query can scope to (or across) versions."""
        cv = str(content_version)
        docs: list[Document] = [
            Document(
                page_content=self._overview_text(overview),
                metadata={
                    "kind": "overview",
                    "content_version": cv,
                    "topic": str(overview.get("topic", "")),
                    "overview_json": json.dumps(overview),
                    "persisted_at": datetime.now().isoformat(),
                },
            )
        ]
        for f in findings or []:
            text = " ".join(
                str(f.get(k, ""))
                for k in ("gap_explanation", "correct_understanding", "student_claim")
            ).strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "kind": "finding",
                        "content_version": cv,
                        "finding_json": json.dumps(f),
                        "persisted_at": datetime.now().isoformat(),
                    },
                )
            )
        self._cache.populate(docs)
        logger.info(
            "Persisted analysis v%s for note %s to vector history (%d doc(s))",
            cv,
            self.note_id,
            len(docs),
        )

    def semantic(self, query: str, k: int = 5) -> list[Document]:
        return self._cache.semantic(query, k)

    def history(self) -> Optional[list[Document]]:
        """Every overview/finding document ever persisted for this note."""
        return self._cache.fetch_all()

    def invalidate(self) -> None:
        self._cache.invalidate()


# ============================================================================
# SQLITE BACKUP CACHE - Future-proofing for analysis history
# ============================================================================


class AnalysisHistoryCache:
    """
    SQLite-based cache for storing analysis history.
    Useful for tracking how analysis changes over versions.
    """

    def __init__(self, bubble_id, note_id, in_memory: bool = False):
        cache_dir = CerebrumPaths().note_analysis_dir(
            bubble_id=bubble_id, note_id=note_id
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        db_path = ":memory:" if in_memory else str(cache_dir / "analysis_history.db")
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id TEXT NOT NULL,
                bubble_id TEXT NOT NULL,
                content_version REAL NOT NULL,
                analysis TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(note_id, content_version, prompt_hash)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_note_version 
            ON analysis_history(note_id, content_version)
            """
        )
        self.conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_analysis(
        self, note_id: str, content_version: float, prompt: str
    ) -> Optional[str]:
        """Get cached analysis for specific version and prompt."""
        prompt_hash = self._hash(prompt)

        row = self.conn.execute(
            """
            SELECT analysis FROM analysis_history
            WHERE note_id=? AND content_version=? AND prompt_hash=?
            """,
            (note_id, content_version, prompt_hash),
        ).fetchone()

        return row[0] if row else None

    def save_analysis(
        self,
        note_id: str,
        bubble_id: str,
        content_version: float,
        analysis: str,
        prompt: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Save analysis to history."""
        prompt_hash = self._hash(prompt)
        metadata_json = json.dumps(metadata) if metadata else None

        self.conn.execute(
            """
            INSERT OR REPLACE INTO analysis_history
            (note_id, bubble_id, content_version, analysis, prompt_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (note_id, bubble_id, content_version, analysis, prompt_hash, metadata_json),
        )
        self.conn.commit()

    def get_version_history(self, note_id: str) -> list[dict]:
        """Get all versions of analysis for a note."""
        rows = self.conn.execute(
            """
            SELECT content_version, analysis, created_at, metadata
            FROM analysis_history
            WHERE note_id=?
            ORDER BY content_version DESC
            """,
            (note_id,),
        ).fetchall()

        return [
            {
                "version": row[0],
                "analysis": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
            }
            for row in rows
        ]

    def close(self):
        self.conn.close()


# ============================================================================
# FLEXIBLE CACHE FACTORY
# ----------------------------------------------------------------------------
# One entry point to construct any cache type. Keeps call sites from hard-
# coding which concrete class backs which kind of cache, and gives a single
# place to register a new cache type later.
# ============================================================================


def make_cache(
    cache_type: CacheType,
    *,
    note_id: str,
    bubble_id: str,
    engram_id: Optional[str] = None,
):
    """Construct the cache for `cache_type`, bound to a note/bubble.

    RETRIEVAL/ENGRAM are semantic (VectorCache-backed); ANALYSIS is the
    deterministic version-keyed JSON cache; HISTORY is the SQLite history
    cache. All keep their existing on-disk locations.
    """
    if cache_type == CacheType.RETRIEVAL:
        return RetrievalCacheInator(note_id=note_id, bubble_id=bubble_id)
    if cache_type == CacheType.ENGRAM:
        return EngramCacheInator(
            note_id=note_id, bubble_id=bubble_id, engram_id=engram_id
        )
    if cache_type == CacheType.ANALYSIS:
        return AnalysisCacheInator(bubble_id=bubble_id, note_id=note_id)
    if cache_type == CacheType.ANALYSIS_VECTOR:
        return AnalysisVectorCacheInator(note_id=note_id, bubble_id=bubble_id)
    if cache_type == CacheType.HISTORY:
        return AnalysisHistoryCache(bubble_id=bubble_id, note_id=note_id)
    raise ValueError(f"Unknown cache type: {cache_type}")
