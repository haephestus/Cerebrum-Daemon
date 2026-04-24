import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths

logger = logging.getLogger(__name__)


# ============================================================================
# ANALYSIS CACHE - Use SQLite (fast, simple, version-based)
# ============================================================================


class AnalysisCacheInator:
    """
    Simple file-based cache for note analysis.
    Cache key: note_id + content_version
    """

    def __init__(self, bubble_id: str, note_id: str):
        self.bubble_id = bubble_id
        self.note_id = note_id
        self.cache_dir = CerebrumPaths().analysis_cache_path(bubble_id)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / note_id

    def get_cached_analysis(self, content_version: float) -> Optional[str]:
        """
        Retrieve cached analysis if it exists for this version.

        Args:
            content_version: Current version of the note

        Returns:
            Cached analysis string or None
        """
        if not self.cache_file.exists():
            return None

        try:
            cache_data = json.loads(self.cache_file.read_text(encoding="utf-8"))

            # Check if cached version matches
            if cache_data.get("content_version") == content_version:
                logger.info(f"Cache HIT for note {self.note_id} v{content_version}")
                return cache_data.get("analysis")

            logger.info(
                f"Cache MISS for note {self.note_id} v{content_version} "
                f"(cached v{cache_data.get('content_version')})"
            )
            return None

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read analysis cache: {e}")
            return None

    def cache_analysis(
        self, content_version: float, analysis: str, metadata: Optional[dict] = None
    ) -> None:
        """
        Store analysis result in cache.

        Args:
            content_version: Version of the note
            analysis: Analysis result to cache
            metadata: Additional metadata to store
        """
        cache_data = {
            "note_id": self.note_id,
            "bubble_id": self.bubble_id,
            "content_version": content_version,
            "analysis": analysis,
            "cached_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")

        logger.info(f"Cached analysis for note {self.note_id} v{content_version}")

    def invalidate_cache(self) -> None:
        """Delete cached analysis for this note."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info(f"Invalidated cache for note {self.note_id}")

    def get_cache_info(self) -> Optional[dict]:
        """Get metadata about cached analysis without loading full content."""
        if not self.cache_file.exists():
            return None

        try:
            cache_data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            return {
                "content_version": cache_data.get("content_version"),
                "cached_at": cache_data.get("cached_at"),
                "metadata": cache_data.get("metadata", {}),
            }
        except (json.JSONDecodeError, KeyError):
            return None


# ============================================================================
# RETRIEVAL CACHE - Use Chroma (semantic search makes sense here)
# ============================================================================


class RetrievalCacheInator:
    """
    Caches retrieved documents from knowledge base.
    Uses Chroma for semantic deduplication and similarity search.
    """

    def __init__(
        self,
        note_id: str,
        bubble_id: str,
    ) -> None:
        self.note_id = note_id
        self.bubble_id = bubble_id
        self.cache_path = CerebrumPaths().analysis_cache_path(bubble_id)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _get_cache(self) -> Chroma:
        """Get or create Chroma collection for this bubble."""
        embedding_model = ConfigManager().load_config().models.embedding_model
        assert embedding_model is not None, "Embedding model not configured"

        return Chroma(
            persist_directory=str(self.cache_path),
            embedding_function=OllamaEmbeddings(model=embedding_model),
            collection_metadata={
                "bubble_id": self.bubble_id,
                "type": "retrieval_cache",
            },
        )

    def cache_populator_inator(
        self,
        retrieved_docs: list[Document] | None,
    ) -> None:
        """
        Cache retrieved documents with metadata.
        """
        if not retrieved_docs:
            logger.warning(f"No documents to cache for note {self.note_id}")
            return

        cached_docs = []
        for doc in retrieved_docs:
            # Preserve original metadata and add cache metadata
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata.update(
                {
                    "note_id": self.note_id,
                    "bubble_id": self.bubble_id,
                    "cached_at": datetime.now().isoformat(),
                }
            )

            cached_docs.append(
                Document(
                    page_content=doc.page_content, metadata=metadata  # Fixed typo!
                )
            )

        try:
            self._get_cache().add_documents(cached_docs)
            logger.info(f"Cached {len(cached_docs)} documents for note {self.note_id}")
        except Exception as e:
            logger.error(f"Failed to cache documents: {e}")

    def deterministic_fetcher(self) -> Optional[list[Document]]:
        """
        Fetch cached documents by exact note_id match.

        Returns:
            List of cached documents or None if not found
        """
        try:
            data = self._get_cache().get(
                where={
                    "$and": [
                        {"note_id": self.note_id},
                        {"bubble_id": self.bubble_id},
                    ]
                }
            )

            if not data["ids"]:
                logger.info(f"No cached docs found for note {self.note_id}")
                return None

            docs = [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(data["documents"], data["metadatas"])
            ]

            logger.info(f"Retrieved {len(docs)} cached docs for note {self.note_id}")
            return docs

        except Exception as e:
            logger.error(f"Failed to fetch cached documents: {e}")
            return None

    def semantic_fetch(self, query: str, k: int = 5) -> list[Document]:
        """
        Fetch similar documents using semantic search.
        Useful for finding related context across notes.

        Args:
            query: Query text for similarity search
            k: Number of results to return

        Returns:
            List of similar documents
        """
        try:
            return self._get_cache().similarity_search(
                query=query, k=k, filter={"bubble_id": self.bubble_id}
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def invalidate_note_cache(self) -> None:
        """Delete cached documents for this specific note."""
        try:
            self._get_cache().delete(
                where={
                    "bubble_id": self.bubble_id,
                    "note_id": self.note_id,
                }
            )
            logger.info(f"Invalidated retrieval cache for note {self.note_id}")
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")

    def invalidate_bubble_cache(self) -> None:
        """Delete entire cache collection for this bubble."""
        try:
            self._get_cache().delete_collection()
            logger.info(f"Deleted entire retrieval cache for bubble {self.bubble_id}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")


# ============================================================================
# SQLITE BACKUP CACHE - Future-proofing for analysis history
# ============================================================================


class AnalysisHistoryCache:
    """
    SQLite-based cache for storing analysis history.
    Useful for tracking how analysis changes over versions.
    """

    def __init__(self, bubble_id, in_memory: bool = False):
        cache_dir = CerebrumPaths().analysis_cache_path(bubble_id)
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
