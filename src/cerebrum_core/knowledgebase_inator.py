import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.markdown_handler_inator import MarkdownChunker
from cerebrum_core.utils.registry.file_chunk_registry_inator import (
    FileChunkRegisterInator,
)

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/cerebrum_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cerebrum")


class KnowledgebaseManager:
    """
    Manages vector store operations independent of specific documents.
    Use this for general queries, listing, and management tasks.
    """

    def __init__(self):
        self.archives_path = CerebrumPaths().kb_archives_path()
        embedding_model = ConfigManager().load_config().models.embedding_model
        if not embedding_model:
            raise ValueError("Embedding model not configured")
        self.embedding_model = embedding_model

    def get_store(
        self,
        collection_name: str,
        domain: str = "default",
        subject: str = "default",
    ) -> Chroma:
        """
        Get a Chroma vector store instance.

        Args:
            collection_name: Name of the collection
            domain: Domain directory
            subject: Subject directory

        Returns:
            Chroma vector store instance
        """
        collection_path = Path(self.archives_path) / domain / subject
        collection_path.mkdir(parents=True, exist_ok=True)

        return Chroma(
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model=self.embedding_model),
            persist_directory=str(collection_path),
        )

    def list_all_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections across all domains and subjects.

        Returns:
            List of dicts with collection info: {domain, subject, collection_name, path, count}
        """
        collections = []
        archives_root = Path(self.archives_path)

        # Traverse domain/subject structure
        for domain_path in archives_root.iterdir():
            if not domain_path.is_dir():
                continue

            for subject_path in domain_path.iterdir():
                if not subject_path.is_dir():
                    continue

                domain = domain_path.name
                subject = subject_path.name

                # Check for chroma.sqlite3 to confirm it's a valid collection
                if (subject_path / "chroma.sqlite3").exists():
                    try:
                        store = self.get_store(subject, domain, subject)
                        count = store._collection.count()

                        collections.append(
                            {
                                "domain": domain,
                                "subject": subject,
                                "collection_name": subject,
                                "path": str(subject_path),
                                "count": count,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to read collection at {subject_path}: {e}"
                        )

        return collections

    def get_collection_info(
        self,
        collection_name: str,
        domain: str = "default",
        subject: str = "default",
    ) -> Dict[str, Any]:
        """
        Get detailed information about a collection.

        Returns:
            Dict with count, collection metadata, and sample documents
        """
        store = self.get_store(collection_name, domain, subject)

        try:
            collection = getattr(store, "_collection", None)
            if collection is None:
                raise RuntimeError("Store has no underlying collection")

            count = collection.count()
            collection_metadata = getattr(collection, "metadata", {}) or {}

            sample_docs: list[dict] = []

            if count > 0:
                results = collection.get(limit=3) or {}

                ids = results.get("ids") or []
                documents = results.get("documents") or []
                metadatas = results.get("metadatas") or []

                for i, doc_id in enumerate(ids):
                    content = documents[i] if i < len(documents) else None
                    metadata = metadatas[i] if i < len(metadatas) else {}

                    sample_docs.append(
                        {
                            "id": doc_id,
                            "content_preview": content[:200] if content else "",
                            "metadata": metadata or {},
                        }
                    )

            return {
                "collection_name": collection_name,
                "domain": domain,
                "subject": subject,
                "count": count,
                "collection_metadata": collection_metadata,
                "sample_documents": sample_docs,
            }

        except Exception as e:
            logger.exception(
                "Failed to get collection info",
                extra={
                    "collection": collection_name,
                    "domain": domain,
                    "subject": subject,
                },
            )
            raise

    def search_across_collections(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple collections.

        Args:
            query: Search query
            domains: List of domains to search (None = all)
            subjects: List of subjects to search (None = all)
            k: Number of results per collection

        Returns:
            List of results with collection info
        """
        all_collections = self.list_all_collections()
        results = []

        for coll_info in all_collections:
            # Filter by domain/subject if specified
            if domains and coll_info["domain"] not in domains:
                continue
            if subjects and coll_info["subject"] not in subjects:
                continue

            try:
                store = self.get_store(
                    coll_info["collection_name"],
                    coll_info["domain"],
                    coll_info["subject"],
                )

                docs = store.similarity_search(query, k=k)

                for doc in docs:
                    results.append(
                        {
                            "domain": coll_info["domain"],
                            "subject": coll_info["subject"],
                            "collection": coll_info["collection_name"],
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to search in {coll_info['domain']}/{coll_info['subject']}: {e}"
                )

        return results

    def delete_collection(
        self,
        collection_name: str,
        domain: str = "default",
        subject: str = "default",
    ) -> None:
        """Delete an entire collection."""
        store = self.get_store(collection_name, domain, subject)
        store.delete_collection()
        logger.info(f"Deleted collection {domain}/{subject}/{collection_name}")

    def delete_by_metadata(
        self,
        collection_name: str,
        metadata_filter: Dict[str, Any],
        domain: str = "default",
        subject: str = "default",
    ) -> int:
        """
        Delete documents matching metadata criteria.

        Args:
            collection_name: Collection name
            metadata_filter: Metadata key-value pairs to match
            domain: Domain directory
            subject: Subject directory

        Returns:
            Number of documents deleted
        """
        store = self.get_store(collection_name, domain, subject)

        # Get all documents with matching metadata
        results = store._collection.get(where=metadata_filter)

        if not results["ids"]:
            logger.info("No documents matched the filter")
            return 0

        # Delete by IDs
        store._collection.delete(ids=results["ids"])
        count = len(results["ids"])
        logger.info(f"Deleted {count} documents matching filter: {metadata_filter}")

        return count

    def get_documents_by_fingerprint(
        self,
        fingerprint: str,
    ) -> List[Dict[str, Any]]:
        """
        Find all documents with a specific fingerprint across all collections.

        Args:
            fingerprint: Document fingerprint to search for

        Returns:
            List of documents with their collection info
        """
        all_collections = self.list_all_collections()
        documents = []

        for coll_info in all_collections:
            try:
                store = self.get_store(
                    coll_info["collection_name"],
                    coll_info["domain"],
                    coll_info["subject"],
                )

                # Search by fingerprint metadata
                results = store._collection.get(where={"fingerprint": fingerprint})

                if results["ids"]:
                    for i, doc_id in enumerate(results["ids"]):
                        documents.append(
                            {
                                "id": doc_id,
                                "domain": coll_info["domain"],
                                "subject": coll_info["subject"],
                                "collection": coll_info["collection_name"],
                                "content": (
                                    results["documents"][i]
                                    if results["documents"]
                                    else ""
                                ),
                                "metadata": (
                                    results["metadatas"][i]
                                    if results["metadatas"]
                                    else {}
                                ),
                            }
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to search in {coll_info['domain']}/{coll_info['subject']}: {e}"
                )

        return documents

    def delete_by_fingerprint_all_collections(self, fingerprint: str) -> int:
        """
        Delete all documents with a specific fingerprint across all collections.

        Args:
            fingerprint: Document fingerprint

        Returns:
            Total number of documents deleted
        """
        total_deleted = 0
        all_collections = self.list_all_collections()

        for coll_info in all_collections:
            try:
                count = self.delete_by_metadata(
                    coll_info["collection_name"],
                    {"fingerprint": fingerprint},
                    coll_info["domain"],
                    coll_info["subject"],
                )
                total_deleted += count
            except Exception as e:
                logger.warning(
                    f"Failed to delete from {coll_info['domain']}/{coll_info['subject']}: {e}"
                )

        logger.info(f"Total deleted across all collections: {total_deleted}")
        return total_deleted


class FileMarkdownChunker(MarkdownChunker):
    """
    Chunks markdown files from knowledgebase
    """

    def __init__(self) -> None:
        super().__init__()
        self.file_chunk_registry = FileChunkRegisterInator()

    def chunk(self, markdown_path: Path, file_fingerprint: str) -> Path:
        markdown_text = markdown_path.read_text(encoding="utf-8")

        annotated_md, registry_rows, _ = self.chunk_markdown(
            markdown_text,
            file_fingerprint=file_fingerprint,
        )

        chunked_path = markdown_path.with_name(markdown_path.stem + ".chunked.md")
        chunked_path.write_text(annotated_md, encoding="utf-8")

        self.file_chunk_registry.register_chunks(registry_rows)
        return chunked_path
