import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.faiss_store_inator import (
    delete_by_metadata as _shared_delete_by_metadata,
)
from cerebrum_core.utils.faiss_store_inator import delete_store as _shared_delete_store
from cerebrum_core.utils.faiss_store_inator import get_or_create_store, iter_docs
from cerebrum_core.utils.faiss_store_inator import save_store as _shared_save_store
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.markdown_handler_inator import MarkdownChunker
from cerebrum_core.utils.database.file_chunk_registry_inator import (
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
    Manages FAISS vector store operations independent of specific documents.
    All actual load/save/iterate/delete logic lives in faiss_store_inator —
    this class is just domain/subject path resolution plus the KB-specific
    operations (cross-collection search, fingerprint lookup) built on top.
    """

    def __init__(self):
        self.archives_path = CerebrumPaths().kb_archives_path()
        embedding_model = ConfigManager().load_config().models.embedding_model
        if not embedding_model:
            raise ValueError("Embedding model not configured")
        self.embedding_model = embedding_model

    def _embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=self.embedding_model)

    def _path(self, domain: str, subject: str) -> Path:
        return Path(self.archives_path) / domain / subject

    def get_store(
        self,
        collection_name: str,
        domain: str = "default",
        subject: str = "default",
    ) -> FAISS:
        """
        Note: unlike Chroma, a FAISS index folder holds exactly one store —
        domain/subject alone determines the path. collection_name no longer
        selects among multiple collections sharing a folder; it's kept for
        call-site/route compatibility with existing callers (all of which
        currently pass subject as collection_name) and logged if it diverges.
        """
        if collection_name != subject:
            logger.warning(
                f"get_store: collection_name={collection_name!r} differs from "
                f"subject={subject!r} — FAISS keys purely off domain/subject, "
                "so collection_name is being ignored here."
            )

        return get_or_create_store(self._path(domain, subject), self._embeddings())

    def save_store(self, store: FAISS, domain: str, subject: str) -> None:
        _shared_save_store(store, self._path(domain, subject))

    def list_all_collections(self) -> List[Dict[str, Any]]:
        collections = []
        archives_root = Path(self.archives_path)

        for domain_path in archives_root.iterdir():
            if not domain_path.is_dir():
                continue

            for subject_path in domain_path.iterdir():
                if not subject_path.is_dir():
                    continue

                domain = domain_path.name
                subject = subject_path.name

                if (subject_path / "index.faiss").exists():
                    try:
                        store = self.get_store(subject, domain, subject)
                        count = len(store.index_to_docstore_id)

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
        store = self.get_store(collection_name, domain, subject)

        try:
            count = len(store.index_to_docstore_id)
            sample_docs: list[dict] = []

            for doc_id, doc in list(iter_docs(store))[:24]:
                sample_docs.append(
                    {
                        "id": doc_id,
                        "content_preview": (
                            doc.page_content if doc.page_content else ""
                        ),
                        "metadata": doc.metadata or {},
                    }
                )

            return {
                "collection_name": collection_name,
                "domain": domain,
                "subject": subject,
                "count": count,
                # FAISS has no collection-level metadata dict the way Chroma
                # does, so this key is intentionally dropped rather than faked.
                "sample_documents": sample_docs,
            }

        except Exception:
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
        all_collections = self.list_all_collections()
        results = []

        for coll_info in all_collections:
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
        _shared_delete_store(self._path(domain, subject))
        logger.info(f"Deleted collection {domain}/{subject}/{collection_name}")

    def delete_by_metadata(
        self,
        collection_name: str,
        metadata_filter: Dict[str, Any],
        domain: str = "default",
        subject: str = "default",
    ) -> int:
        store = self.get_store(collection_name, domain, subject)
        count = _shared_delete_by_metadata(
            store, self._path(domain, subject), metadata_filter
        )
        if count:
            logger.info(f"Deleted {count} documents matching filter: {metadata_filter}")
        else:
            logger.info("No documents matched the filter")
        return count

    def get_documents_by_fingerprint(self, fingerprint: str) -> List[Dict[str, Any]]:
        all_collections = self.list_all_collections()
        documents = []

        for coll_info in all_collections:
            try:
                store = self.get_store(
                    coll_info["collection_name"],
                    coll_info["domain"],
                    coll_info["subject"],
                )

                for doc_id, doc in iter_docs(store):
                    # matches EmbeddInator.doc_metadata's "file_fingerprint" key
                    if (doc.metadata or {}).get("file_fingerprint") == fingerprint:
                        documents.append(
                            {
                                "id": doc_id,
                                "domain": coll_info["domain"],
                                "subject": coll_info["subject"],
                                "collection": coll_info["collection_name"],
                                "content": doc.page_content,
                                "metadata": doc.metadata or {},
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to search in {coll_info['domain']}/{coll_info['subject']}: {e}"
                )

        return documents

    def delete_by_fingerprint_all_collections(self, fingerprint: str) -> int:
        total_deleted = 0
        all_collections = self.list_all_collections()

        for coll_info in all_collections:
            try:
                count = self.delete_by_metadata(
                    coll_info["collection_name"],
                    {"file_fingerprint": fingerprint},
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

        offsets_path = markdown_path.with_suffix("").with_suffix(".pageoffsets.json")
        page_offsets = None
        if offsets_path.exists():
            raw = json.loads(offsets_path.read_text(encoding="utf-8"))
            page_offsets = [tuple(row) for row in raw]
        else:
            logger.warning(
                f"No page-offset sidecar found at {offsets_path} — pdf_page fields will be null"
            )

        annotated_md, registry_rows, _ = self.chunk_markdown(
            markdown_text,
            file_fingerprint=file_fingerprint,
            page_offsets=page_offsets,
        )

        chunked_path = markdown_path.with_name(markdown_path.stem + ".chunked.md")
        chunked_path.write_text(annotated_md, encoding="utf-8")

        self.file_chunk_registry.register_chunks(registry_rows)
        return chunked_path
