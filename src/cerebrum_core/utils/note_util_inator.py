import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from langchain_core.documents.base import Document

from cerebrum_core.model_inator import NoteContent, NoteStorage
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.markdown_handler_inator import MarkdownChunker
from cerebrum_core.database.note_chunk_registry_inator import (
    NoteChunkRegisterInator,
)


# convert the notes to markdown
# chunk notes and register chunks
def note_processor_inator(bubble_id: str, note_id: str, note_content: NoteContent):
    flattened_note = NoteToMarkdownInator().flatten(note_content)
    NoteChunkerInator().chunk(
        bubble_id=bubble_id, note_id=note_id, flattened_note=flattened_note
    )
    logging.info(f"Note: {note_id} processed successfully")


def diff_collapser_inator(note: NoteStorage) -> NoteStorage:
    """
    Cleans up note diffs and prevents markdonw ballooning
    """
    # load note into memory
    history = note.history.content
    if len(history) <= 1:
        return note

    latest_by_version = {}
    version_order = []

    for entry in history:
        ver = entry.version
        if ver not in latest_by_version:
            version_order.append(ver)
        latest_by_version[ver] = entry

    note.history.content = [latest_by_version[v] for v in version_order]
    return note


# ------------------------------ NOTE PACKAGING ------------------------------ #
#
# Each note is now a folder, not a single file:
#
#   notes/<note_id>/content.json   title, content, metadata, history —
#                                   everything except the ink blob.
#                                   metadata.ink_hash/ink_version stay
#                                   here; they're cheap scalars.
#   notes/<note_id>/ink.json       just `{"ink": [...]}`.
#
# Both are plain, human-readable JSON — no compression, no archive
# format. `cat`/open-in-editor either one directly.
#
# The `filename` identifier used everywhere else (URL params,
# NoteOut.filename, the Flutter client, note_registry) is UNCHANGED —
# still `<note_id>.json` as a string. Internally, that string is only
# ever used to derive the folder name (stripping ".json"); nothing
# outside this section needs to know storage moved from a file to a
# folder.
#
# Legacy notes (flat `<note_id>.json` files from before this existed)
# are handled transparently: `_load_note`/`_load_note_skip_ink` check
# for the folder form first, and fall back to reading the old flat file
# if no folder exists yet. `_save_note` always writes folder form and
# deletes the old flat file if migrating one — so any legacy note
# self-migrates the next time it's saved for any reason.
#
# Not handled: a crash exactly mid-migration (folder partially written,
# old flat file not yet cleaned up, or vice versa). `_load_note` prefers
# the folder if its content.json exists, which covers the common case,
# but this isn't a transactional guarantee — acceptable for a local
# single-user note store, worth knowing if that assumption ever changes.
#
# IMPORTANT: `_load_note` is the ONLY sanctioned way to read a note's
# storage off disk. Do not `.read_text()` a note path directly anywhere
# else in the codebase — a note's on-disk shape (flat file vs folder)
# is an implementation detail this module hides. Any caller that
# bypasses it and reads a path itself will silently break the next time
# that shape changes, exactly as `active_analysis` in
# learning_center_inator.py did when notes moved to folder form.


def _note_dir(notes_dir: Path, filename: str) -> Path:
    stem = filename[:-5] if filename.endswith(".json") else filename
    return notes_dir / stem


def _legacy_flat_filepath(notes_dir: Path, filename: str) -> Path:
    return notes_dir / filename


def _note_exists(notes_dir: Path, filename: str) -> bool:
    if (_note_dir(notes_dir, filename) / "content.json").exists():
        return True
    return _legacy_flat_filepath(notes_dir, filename).exists()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write-to-temp-then-replace, so a crash mid-write can't leave a
    half-written JSON file behind."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _load_note(notes_dir: Path, filename: str) -> NoteStorage:
    """Load a note, merging in ink.json if the note is in folder form."""
    note_dir = _note_dir(notes_dir, filename)
    content_path = note_dir / "content.json"

    if content_path.exists():
        stored_data = json.loads(content_path.read_text(encoding="utf-8"))
        ink_path = note_dir / "ink.json"
        if ink_path.exists():
            ink_data = json.loads(ink_path.read_text(encoding="utf-8"))
            stored_data["ink"] = ink_data.get("ink", [])
        else:
            stored_data.setdefault("ink", [])
    else:
        # Legacy flat file. Whatever "ink" is embedded (or [] if this
        # note predates ink entirely) is used as-is — _save_note
        # migrates it into folder form on next save.
        legacy_path = _legacy_flat_filepath(notes_dir, filename)
        stored_data = json.loads(legacy_path.read_text(encoding="utf-8"))

    return NoteStorage(**stored_data)


def _load_note_skip_ink(notes_dir: Path, filename: str) -> Dict[str, Any]:
    """For listing: reads only content.json, whether the note is in
    folder form or (still) a legacy flat file."""
    content_path = _note_dir(notes_dir, filename) / "content.json"
    if content_path.exists():
        return json.loads(content_path.read_text(encoding="utf-8"))

    legacy_path = _legacy_flat_filepath(notes_dir, filename)
    return json.loads(legacy_path.read_text(encoding="utf-8"))


def _note_needs_ink_migration(notes_dir: Path, filename: str) -> bool:
    """True if ink.json doesn't exist yet for this note — covers both a
    brand-new folder-form note before its first ink write and a legacy
    flat file being migrated for the first time."""
    ink_path = _note_dir(notes_dir, filename) / "ink.json"
    return not ink_path.exists()


def _save_note(
    notes_dir: Path,
    filename: str,
    stored_note: NoteStorage,
    *,
    write_ink: bool = True,
) -> None:
    """
    Writes content.json unconditionally (cheap — no ink in it). Writes
    ink.json only when `write_ink=True`; when False, ink.json is simply
    left untouched — no read, no copy, no-op, since these are plain
    sibling files rather than an archive that has to be rewritten whole.
    If this call is migrating a legacy flat file, the old file is
    removed once the folder form is fully written.
    """
    note_dir = _note_dir(notes_dir, filename)
    note_dir.mkdir(parents=True, exist_ok=True)

    content_path = note_dir / "content.json"
    _atomic_write_text(
        content_path,
        stored_note.model_dump_json(indent=2, exclude={"ink"}),
    )

    if write_ink:
        ink_path = note_dir / "ink.json"
        _atomic_write_text(
            ink_path,
            json.dumps({"ink": stored_note.ink}, indent=2),
        )

    legacy_path = _legacy_flat_filepath(notes_dir, filename)
    if legacy_path.exists() and legacy_path.is_file():
        legacy_path.unlink()


def _delete_note_files(notes_dir: Path, filename: str) -> None:
    note_dir = _note_dir(notes_dir, filename)
    if note_dir.exists() and note_dir.is_dir():
        shutil.rmtree(note_dir)

    legacy_path = _legacy_flat_filepath(notes_dir, filename)
    legacy_path.unlink(missing_ok=True)


# modify to allow for chunking and chunk by chunk analysis
class NoteToMarkdownInator:
    """
    Converts an AppFlowy-style note into markdown
    """

    def __init__(self, convert_tables: bool = True) -> None:
        self.convert_tables = convert_tables

    # ------------ Core Public Method ------------- #
    def flatten(self, note: NoteContent) -> str:
        children = note.document["children"]
        lines = []

        for block in children:
            block_id = block.get("id", "")  # AppFlowy block ID
            handler = getattr(self, f"_handle_{block['type'].replace('/','_')}", None)
            if handler:
                result = handler(block)
                if result:
                    if block_id:
                        lines.append(f"<!-- block_id:{block_id} -->")
                    lines.append(result)
            lines.append("")

        return "\n".join(lines).strip()

    # ------------ Block Handlers --------------#
    def _handle_heading(self, block):
        level = block["data"]["level"]
        text = self._extract_text(block)
        return f"{'#' * level} {text}"

    def _handle_paragraph(self, block):
        text = self._extract_text(block)
        return text.strip() if text else None

    def _handle_divider(self):
        return "---"

    def _handle_table(self, block):
        if not self.convert_tables:
            return "[TABLE OMITTED]"

        return self._flatten_table(block)

    # ---------------- Helpers -----------------#
    def _extract_text(self, block):
        """Extracts linear text from delta[].insert"""
        if not block:
            return ""

        delta = block.get("data", {}).get("delta", [])
        text = ""
        for item in delta:
            if isinstance(item, dict):
                text += item.get("insert", "")
            elif isinstance(item, str):
                text += item
        return text

    def _flatten_table(self, table_block):
        """Converts Appflowy table -> markdown table."""
        rows = table_block["data"]["rowsLen"]
        cols = table_block["data"]["colsLen"]
        cells = table_block.get("children", [])

        matrix = [["" for _ in range(cols)] for _ in range(rows)]

        for cell in cells:
            data = cell.get("data", [])
            row = data.get("rowPosition")
            col = data.get("colPosition")

            # Defensive checks
            if row is None or col is None:
                continue
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue

            inner = cell["children"][0] if cell.get("children") else None
            matrix[row][col] = self._extract_text(inner)

        md = []
        md.append("| " + " | ".join(matrix[0]) + " |")
        md.append("| " + " | ".join(["---"] * cols) + " |")

        for row in matrix[1:]:
            md.append("| " + " | ".join(row) + " |")

        return "\n".join(md)


class NoteChunkerInator(MarkdownChunker):
    """
    Chunks notes converted to markdown and registers chunks for analysis.
    Ensures that LangChain Document objects carry accurate structural metadata
    for pure vector search retrieval without data string pollution.
    """

    def __init__(self, generate_artifacts: bool = True):
        super().__init__()
        self.note_chunk_registry = NoteChunkRegisterInator()
        self.generate_artifacts = generate_artifacts

    def chunk(
        self, flattened_note: str, note_id: str, bubble_id: str
    ) -> tuple[str, list[Document]]:
        # chunk_markdown returns a 3-tuple: (annotated_md, registry_rows, documents)
        # The documents list now contains fully hydrated metadata dicts (byte_start, byte_end, etc.)
        annotated_md, registry_rows, documents = self.chunk_markdown(
            flattened_note, note_id=note_id
        )

        if self.generate_artifacts:
            chunked_path = CerebrumPaths().chunked_note_path(
                bubble_id=bubble_id, note_id=note_id
            )
            chunked_path.parent.mkdir(parents=True, exist_ok=True)
            chunked_path = chunked_path.parent / f"{note_id}.md"
            chunked_path.write_text(annotated_md, encoding="utf-8")
            logging.info(f"Note: {note_id} chunked successfully and saved to disk")

        # Sync structure to SQL/Dataframe analyzer layout tracking rows
        self.note_chunk_registry.register_chunks(registry_rows)
        logging.info(f"Note: {note_id} registered successfully to tracking store")

        return annotated_md, documents
