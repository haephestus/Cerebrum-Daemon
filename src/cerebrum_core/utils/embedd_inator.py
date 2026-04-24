import hashlib
import logging
import os
from pathlib import Path

from langchain_core.documents import Document

from cerebrum_core.knowledgebase_inator import KnowledgebaseManager
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


class EmbeddInator:
    """
    Handles embedding operations for specific documents.
    Use this when you have a specific document to embed.
    """

    def __init__(self, original_name: str, file_fingerprint: str):
        self.original_name = original_name
        self.file_fingerprint = file_fingerprint
        self.registry = FileChunkRegisterInator()
        self.vector_manager = KnowledgebaseManager()

    def embed_from_chunked_markdown(
        self,
        chunked_markdown: Path,
        collection_name: str,
        domain: str = "default",
        subject: str = "default",
    ) -> None:
        """
        Embed chunks from .chunked.md file.

        Args:
            chunked_markdown: Path to .chunked.md file
            collection_name: Collection name
            domain: Domain directory
            subject: Subject directory
        """
        # Check progress
        progress = self.registry.get_embedding_progress(self.file_fingerprint)

        if progress["total"] == 0:
            logger.warning("No chunks registered - aborting")
            return

        if progress["remaining"] == 0:
            logger.info("All chunks already embedded")
            return

        logger.info(
            f"Resuming: {progress['completed']}/{progress['total']} completed "
            f"({progress['progress_pct']:.1f}%)"
        )

        # Get unembedded chunks
        unembedded = self.registry.get_unembedded_chunks(self.file_fingerprint)

        # Read file once
        with open(chunked_markdown, "rb") as f:
            file_bytes = f.read()

        # Get vector store
        store = self.vector_manager.get_store(collection_name, domain, subject)

        # Embed each chunk
        for record in unembedded:
            chunk_bytes = file_bytes[record.byte_start : record.byte_end]
            chunk_content = chunk_bytes.decode("utf-8")
            chunk_fingerprint = hashlib.sha256(
                chunk_content.encode("utf-8")
            ).hexdigest()

            doc_metadata = {
                "original_name": self.original_name,
                "file_fingerprint": self.file_fingerprint,
                "chunk_fingerprint": chunk_fingerprint,
                "chunk_index": record.chunk_index,
                "chunk_type": record.chunk_type,
                "parent_chunk_index": record.parent_chunk_index or "",
                "domain": domain,
                "subject": subject,
            }

            doc = Document(page_content=chunk_content, metadata=doc_metadata)

            try:
                store.add_documents([doc])
                self.registry.mark_embedded(self.file_fingerprint, chunk_fingerprint)
                logger.info(f"Embedded chunk {record.chunk_index}")
            except Exception as e:
                logger.error(f"Failed at chunk {record.chunk_index}: {e}")
                raise

        logger.info("Embedding complete")

    def delete_embedded_document(self) -> int:
        """
        Delete all chunks for this document from all collections.

        Returns:
            Number of documents deleted
        """
        return self.vector_manager.delete_by_fingerprint_all_collections(
            self.file_fingerprint
        )
