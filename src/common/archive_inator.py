import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from models.model_inator import Note
from cerebrum_core.user_inator import ConfigManager
from vectorstore.embeddings_inator import get_embeddings
from vectorstore.faiss_store_inator import (
    delete_store,
    get_or_create_store,
    iter_docs,
    save_store,
)
from common.file_util_inator import CerebrumPaths

logger = logging.getLogger(__name__)


class AnalysisArchiveInator:
    """
    Adds historical note versions, on a chunk by chunk basis in order
    to archive the note for analysis, and progress monitoring.
    """

    def __init__(
        self,
        note: Note,
        archives_path: str,
        chunks: Optional[list[Document]] = None,
    ) -> None:
        self.note = note
        self.archives_path = archives_path
        self.chunks = chunks

    def _note_dir(self) -> Path:
        # One FAISS index folder per note, unlike Chroma where every note's
        # archive was a named collection sharing archives_path.
        return Path(self.archives_path) / self.note.note_id

    def _embeddings(self) -> OllamaEmbeddings:
        return get_embeddings()

    def archive_init_inator(self) -> None:
        """Stores snapshots of notes in a historic database."""
        self._get_archives()

    def archive_populator_inator(self) -> dict:
        """
        Add note chunks to the archive.

        Deterministic per-chunk ids (chunk_id + fingerprint) mean an
        unchanged chunk re-submitted for archiving is a no-op instead of
        a duplicate insert — only genuinely new/changed chunk content
        gets added.
        """
        assert self.chunks is not None

        store = self._get_archives()
        existing_ids = set(store.index_to_docstore_id.values())

        now = datetime.now(timezone.utc).isoformat()

        to_add: list[Document] = []
        add_ids: list[str] = []
        skipped: list[str] = []

        for chunk in self.chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            assert chunk_id is not None
            fingerprint = chunk.metadata.get("fingerprint")
            doc_id = f"{chunk_id}:{fingerprint}"

            if doc_id in existing_ids:
                skipped.append(chunk_id)
                continue

            to_add.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "note_id": chunk.metadata.get("note_id"),
                        "chunk_id": chunk_id,
                        "fingerprint": fingerprint,
                        "content_version": chunk.metadata.get("content_version"),
                        "header": chunk.metadata.get("header", ""),
                        "header_level": chunk.metadata.get("header_level"),
                        "archived_at": now,
                    },
                )
            )
            add_ids.append(doc_id)

        if to_add:
            store.add_documents(to_add, ids=add_ids)
            save_store(store, self._note_dir())

        logger.info(
            f"[ARCHIVE] note {self.note.note_id}: "
            f"{len(to_add)} chunk(s) added, {len(skipped)} unchanged/skipped"
        )

        return {"added": len(to_add), "skipped": len(skipped)}

    def archive_cleaner_inator(self) -> None:
        """DANGER: Deletes entire collection(note)"""
        try:
            delete_store(self._note_dir())
            logger.info(f"Deleted collection: {self.note.note_id}")
        except Exception as e:
            logger.warning(f"Collection not found or error: {self.note.note_id} - {e}")

    def archive_browser_inator(self, bubble_id) -> dict | None:
        """
        Read-only browse of this note's archived chunk history.
        (Return shape unchanged — see original docstring.)
        """
        note_file = (
            CerebrumPaths().note_root_dir(bubble_id) / f"{self.note.note_id}.json"
        )

        if not self._note_dir().exists():
            return None

        if not note_file.exists():
            logger.warning(
                f"No note file: {self.note.note_id}.json found for bubble: {bubble_id}"
            )

        store = self._get_archives_readonly()
        if store is None:
            return None

        docs = list(iter_docs(store))
        if not docs:
            return None

        chunks: dict[str, list[dict]] = {}

        for _doc_id, doc in docs:
            metadata = doc.metadata or {}
            chunk_id = metadata.get("chunk_id", "unknown_chunk")

            raw_version = metadata.get("content_version")
            if isinstance(raw_version, (str, int, float)):
                content_version = float(raw_version)
            else:
                logger.warning(
                    f"Missing/invalid content_version in archive metadata for "
                    f"note {self.note.note_id} chunk {chunk_id}: {raw_version!r}"
                )
                content_version = 0.0

            entry = {
                "content_version": content_version,
                "fingerprint": metadata.get("fingerprint"),
                "header": metadata.get("header", ""),
                "header_level": metadata.get("header_level"),
                "archived_at": metadata.get("archived_at"),
                "content": doc.page_content,
            }
            chunks.setdefault(str(chunk_id), []).append(entry)

        for chunk_id in chunks:
            chunks[chunk_id].sort(key=lambda e: e["content_version"])

        return {
            self.note.note_id: {
                "filename": note_file.name,
                "note_name": self.note.title,
                "chunk_count": len(chunks),
                "entry_count": len(docs),
                "chunks": chunks,
            }
        }

    def _get_archives_readonly(self):
        """Read-only load — returns None if this note has no archive yet."""
        note_dir = self._note_dir()
        if not (note_dir / "index.faiss").exists():
            logger.info(
                f"No archived collection for note {self.note.note_id} in {note_dir}"
            )
            return None
        return get_or_create_store(note_dir, self._embeddings())

    def _get_archives(self) -> FAISS:
        """Write path — creates the index if missing."""
        assert self.note is not None
        return get_or_create_store(self._note_dir(), self._embeddings())


def list_archived_note_ids(archives_path: str) -> list[str]:
    """
    Bubble-wide browsing: list every note_id that has an archived
    FAISS index under this bubble's archive path (one subfolder per note).
    """
    root = Path(archives_path)
    if not root.exists():
        return []
    return [
        p.name for p in root.iterdir() if p.is_dir() and (p / "index.faiss").exists()
    ]
