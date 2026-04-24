from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from cerebrum_core.model_inator import ArchivedNote, ArchivedNoteContent, NoteStorage
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths


class AnalysisArchiveInator:
    """
    Adds historical note versions, on a chunk by chunk basis in order
    to archive the note for analysis, and progress monitoring
    """

    def __init__(
        self,
        note: NoteStorage,
        archives_path: str,
        chunks: Optional[list[Document]] = None,
    ) -> None:
        self.note = note
        self.archives_path = archives_path
        self.chunks = chunks

    def archive_init_inator(self) -> None:
        """
        Stores snapshots of notes in a historic database
        """
        self._get_archives()

    def archive_populator_inator(self) -> None:
        """
        Add note chunks to the archive
        """

        assert self.chunks is not None
        # pass chunk object
        # chunk notes
        note = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    "note_id": chunk.metadata.get("note_id"),
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "fingerprint": chunk.metadata.get("fingerprint"),
                    "generated_at": chunk.metadata.get("generated_at"),
                    "header_level": chunk.metadata.get("header_level"),
                    "content_version": chunk.metadata.get("content_version"),
                },
            )
            for chunk in self.chunks
        ]

        self._get_archives().add_documents(note)

    def archive_cleaner_inator(self) -> None:
        """
        DANGER: Deletes entire collection(note)
        """
        try:
            self._get_archives().delete_collection()
            print(f"Deleted collection: {self.note.note_id}")

        except Exception as e:
            print(f"Collection not found or error: {self.note.note_id} - {e}")

    def archive_browser_inator(self, bubble_id) -> dict | None:
        note_file = (
            CerebrumPaths().note_root_dir(bubble_id) / f"{self.note.note_id}.json"
        )

        if not Path(self.archives_path).exists():
            return None

        if not note_file.exists():
            print(
                f" \n No note: {self.note.note_id}.json found for bubble: {bubble_id}",
            )

        raw_data = self._get_archives().get()

        versions = []
        for doc_content, metadata in zip(raw_data["documents"], raw_data["metadatas"]):
            version = metadata.get("version", self.note.metadata.content_version)
            versions.append(
                ArchivedNoteContent(
                    version=float(version),
                    content=doc_content,
                )
            )

        versions.sort(key=lambda x: x.version)

        historical_note = ArchivedNote(
            note_id=self.note.note_id,
            note_name=self.note.title,
            versions=versions,
        )

        return {"filename": note_file.name, "archive": historical_note}

    def _get_archives(self) -> Chroma:
        """Helper: Get Chroma archive instance from disk"""
        embedding_model = ConfigManager().load_config().models.embedding_model
        # TODO: find a better alternative than assert
        assert embedding_model is not None
        assert self.note is not None

        # embedd notes
        return Chroma(
            collection_name=self.note.note_id,
            embedding_function=OllamaEmbeddings(model=embedding_model),
            create_collection_if_not_exists=True,
            persist_directory=str(self.archives_path),
            collection_metadata={
                "note_title": self.note.title,
                "note_id": self.note.note_id,
                "bubble_id": self.note.bubble_id,
                # "bubble_name": self.note.bubble_name
            },
        )
