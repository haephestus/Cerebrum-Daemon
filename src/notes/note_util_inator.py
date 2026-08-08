import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from langchain_core.documents.base import Document

from models.model_inator import NoteContent, NoteStorage, Page, PageMetadata
from common.file_util_inator import CerebrumPaths
from notes.markdown_handler_inator import MarkdownChunker
from notes.block_chunker_inator import Block, pack_blocks
from database.note_chunk_registry_inator import (
    NoteChunkRegisterInator,
)


# convert the notes to markdown
# chunk notes and register chunks
def note_processor_inator(bubble_id: str, note_id: str, note_content: NoteContent):
    # Block-aligned chunking (gap 1 / stream A): packs whole blocks, registers
    # note_chunks + the chunk↔block join. Replaces the old overlap-based path.
    NoteChunkerInator().chunk_note(
        note=note_content, note_id=note_id, bubble_id=bubble_id
    )
    logging.info(f"Note: {note_id} processed successfully")


def note_pages(note: NoteStorage) -> list[Page]:
    """View ANY note as pages. If it already has pages, return them ordered;
    otherwise synthesise a single page from the legacy content/ink/history/
    metadata — so page-aware code (chunking, analysis, sync) works uniformly
    without forcing every note to migrate at once."""
    if note.pages:
        return sorted(note.pages, key=lambda p: p.page_index)
    return [
        Page(
            page_id=f"{note.note_id or 'note'}-p0",
            page_index=0,
            document=(note.content.document if note.content else {}),
            ink=list(note.ink or []),
            history=note.history,
            metadata=PageMetadata(
                content_hash=note.metadata.content_hash,
                content_version=note.metadata.content_version,
                ink_hash=note.metadata.ink_hash,
                ink_version=note.metadata.ink_version,
                version_vector=dict(note.metadata.version_vector),
                last_modified=note.metadata.last_modified,
            ),
        )
    ]


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
    note_dir = _note_dir(notes_dir, filename)
    if (note_dir / "manifest.json").exists():  # new per-page folder layout
        return True
    if (note_dir / "content.json").exists():  # previous single-file layout
        return True
    return _legacy_flat_filepath(notes_dir, filename).exists()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write-to-temp-then-replace, so a crash mid-write can't leave a
    half-written JSON file behind."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _load_note(notes_dir: Path, filename: str) -> NoteStorage:
    """Load a note. Prefers the new per-page folder layout (manifest.json +
    pages/<id>/{content,ink,history}.json); falls back to the previous single
    content.json/ink.json, then to a legacy flat file. `content`/`ink` are
    mirrored from page 0 so callers that still read the flat fields keep working
    while `pages` carries the structured truth."""
    note_dir = _note_dir(notes_dir, filename)
    manifest_path = note_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages_root = note_dir / "pages"
        page_dicts: list[dict] = []
        for pid in manifest.get("page_order", []):
            cpath = pages_root / pid / "content.json"
            if not cpath.exists():
                continue
            cdata = json.loads(cpath.read_text(encoding="utf-8"))
            ipath = pages_root / pid / "ink.json"
            ink = (
                json.loads(ipath.read_text(encoding="utf-8")).get("ink", [])
                if ipath.exists()
                else []
            )
            hpath = pages_root / pid / "history.json"
            hist = json.loads(hpath.read_text(encoding="utf-8")) if hpath.exists() else {}
            page_dicts.append(
                {
                    "page_id": cdata.get("page_id", pid),
                    "page_index": cdata.get("page_index", 0),
                    "document": cdata.get("document", {}),
                    "ink": ink,
                    "metadata": cdata.get("metadata", {}),
                    "history": hist,
                }
            )
        first = page_dicts[0] if page_dicts else {"document": {}, "ink": []}
        return NoteStorage(
            title=manifest.get("title", ""),
            note_id=manifest.get("note_id", ""),
            bubble_id=manifest.get("bubble_id", ""),
            analyse_note=manifest.get("analyse_note", True),
            content=NoteContent(document=first.get("document", {})),
            ink=first.get("ink", []),
            metadata=manifest.get("metadata", {}),
            history=manifest.get("history", {}),
            pages=page_dicts,
        )

    content_path = note_dir / "content.json"
    if content_path.exists():  # previous single-file layout (may carry `pages`)
        stored_data = json.loads(content_path.read_text(encoding="utf-8"))
        ink_path = note_dir / "ink.json"
        stored_data["ink"] = (
            json.loads(ink_path.read_text(encoding="utf-8")).get("ink", [])
            if ink_path.exists()
            else stored_data.get("ink", [])
        )
        return NoteStorage(**stored_data)

    # Legacy flat file.
    legacy_path = _legacy_flat_filepath(notes_dir, filename)
    stored_data = json.loads(legacy_path.read_text(encoding="utf-8"))
    return NoteStorage(**stored_data)


def _load_note_skip_ink(notes_dir: Path, filename: str) -> Dict[str, Any]:
    """For listing: note-level fields + page-0 document, no ink. Handles the
    per-page folder layout, the previous content.json, and legacy flat files."""
    note_dir = _note_dir(notes_dir, filename)
    manifest_path = note_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        doc: Dict[str, Any] = {}
        order = manifest.get("page_order", [])
        if order:
            cpath = note_dir / "pages" / order[0] / "content.json"
            if cpath.exists():
                doc = json.loads(cpath.read_text(encoding="utf-8")).get("document", {})
        return {
            "title": manifest.get("title", ""),
            "note_id": manifest.get("note_id", ""),
            "bubble_id": manifest.get("bubble_id", ""),
            "analyse_note": manifest.get("analyse_note", True),
            "metadata": manifest.get("metadata", {}),
            "content": {"document": doc},
        }

    content_path = note_dir / "content.json"
    if content_path.exists():
        return json.loads(content_path.read_text(encoding="utf-8"))

    legacy_path = _legacy_flat_filepath(notes_dir, filename)
    return json.loads(legacy_path.read_text(encoding="utf-8"))


def _note_needs_ink_migration(notes_dir: Path, filename: str) -> bool:
    """True if this note has no ink file yet — page-0 ink.json in the folder
    layout, or the top-level ink.json in the old layout."""
    note_dir = _note_dir(notes_dir, filename)
    manifest_path = note_dir / "manifest.json"
    if manifest_path.exists():
        order = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "page_order", []
        )
        if not order:
            return True
        return not (note_dir / "pages" / order[0] / "ink.json").exists()
    return not (note_dir / "ink.json").exists()


def _save_note(
    notes_dir: Path,
    filename: str,
    stored_note: NoteStorage,
    *,
    write_ink: bool = True,
) -> None:
    """
    Writes the per-page folder layout:

        notes/<id>/manifest.json            note-level: title, ids, metadata,
                                            note history, ordered page_order
        notes/<id>/pages/<page_id>/
            content.json                    page document + page metadata
            ink.json                        {"ink": [...]}  (skipped if write_ink=False)
            history.json                    page diff history

    Any note is viewed as pages via note_pages() (a legacy single-document note
    yields one synthesised page), so create/update/sync all produce this shape.
    Page folders no longer present are removed; the previous content.json/ink.json
    and any legacy flat file are retired once the folder form is written.
    """
    note_dir = _note_dir(notes_dir, filename)
    pages_root = note_dir / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)

    pages = note_pages(stored_note)

    manifest = {
        "title": stored_note.title,
        "note_id": stored_note.note_id,
        "bubble_id": stored_note.bubble_id,
        "analyse_note": stored_note.analyse_note,
        "metadata": json.loads(stored_note.metadata.model_dump_json()),
        "history": json.loads(stored_note.history.model_dump_json()),
        "page_order": [p.page_id for p in pages],
    }
    _atomic_write_text(note_dir / "manifest.json", json.dumps(manifest, indent=2))

    keep: set[str] = set()
    for p in pages:
        pdir = pages_root / p.page_id
        pdir.mkdir(parents=True, exist_ok=True)
        keep.add(p.page_id)
        _atomic_write_text(
            pdir / "content.json",
            json.dumps(
                {
                    "page_id": p.page_id,
                    "page_index": p.page_index,
                    "document": p.document,
                    "metadata": json.loads(p.metadata.model_dump_json()),
                },
                indent=2,
            ),
        )
        if write_ink:
            _atomic_write_text(pdir / "ink.json", json.dumps({"ink": p.ink}, indent=2))
        _atomic_write_text(pdir / "history.json", p.history.model_dump_json(indent=2))

    # Drop page folders for pages that were removed.
    for existing in pages_root.iterdir():
        if existing.is_dir() and existing.name not in keep:
            shutil.rmtree(existing, ignore_errors=True)

    # Retire the previous single-file layout + any legacy flat file.
    (note_dir / "content.json").unlink(missing_ok=True)
    if write_ink:
        (note_dir / "ink.json").unlink(missing_ok=True)
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

    # ------------ Structured (block-aligned) view ------------- #
    def flatten_blocks(self, note: NoteContent) -> list[Block]:
        """Per-block (id, type, rendered markdown) — the structured parallel to
        flatten(), used by block-aligned chunking. Blocks that render empty are
        dropped; a missing block id gets an index fallback so no content is
        silently lost (it just can't be block-anchored precisely)."""
        out: list[Block] = []
        for i, block in enumerate(note.document.get("children", [])):
            btype = block.get("type", "")
            handler = getattr(self, f"_handle_{btype.replace('/', '_')}", None)
            if not handler:
                continue
            try:
                result = handler(block)
            except TypeError:
                result = handler()  # e.g. _handle_divider takes no block arg
            if not result:
                continue
            out.append(
                Block(block_id=block.get("id") or f"blk{i}", block_type=btype, text=result)
            )
        return out

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

    @staticmethod
    def _pages_for_chunking(note) -> list[tuple[str, dict]]:
        """(page_id, document) per page — from a NoteStorage's pages, or one
        synthesised page for a legacy NoteContent (single document)."""
        if isinstance(note, NoteStorage):
            return [(p.page_id, p.document) for p in note_pages(note)]
        if isinstance(note, NoteContent):
            return [("p0", note.document)]
        return [("p0", getattr(note, "document", {}) or {})]

    def chunk_note(
        self, note, note_id: str, bubble_id: str
    ) -> tuple[str, list[Document]]:
        """Block-aligned chunking (gap 1). Chunks EACH PAGE separately so a chunk
        never crosses a page boundary (page > chunk > block), with a running
        global chunk_index and byte offsets into one assembled .md (byte-offset
        addressable exactly like before, so the analyser's chunk_fetcher keeps
        working). Registers `note_chunks` + the `note_chunk_blocks` join.

        `note` may be a NoteStorage (multi-page) or a NoteContent (single doc);
        `_pages_for_chunking` yields one synthesised page for the legacy shape.
        """
        pages = self._pages_for_chunking(note)
        md = NoteToMarkdownInator()

        parts: list[str] = []
        registry_rows: list[tuple] = []
        block_rows: list[tuple] = []
        documents: list[Document] = []
        cursor = 0  # byte offset into the assembled .md (UTF-8)
        chunk_index = 0  # running across ALL pages

        for page_id, document in pages:
            blocks = md.flatten_blocks(NoteContent(document=document or {}))
            plans = pack_blocks(blocks, max_tokens=512, token_len=self._token_count)
            for plan in plans:
                fp = self._chunk_fingerprint(plan.text)
                annotation = (
                    f"<!-- CHUNK_START chunk_index:{chunk_index} "
                    f"page:{page_id} fingerprint:{fp} -->\n"
                )
                head = annotation + plan.text
                piece = head + "\n\n"
                byte_start = cursor
                byte_end = cursor + len(head.encode("utf-8"))
                cursor += len(piece.encode("utf-8"))
                parts.append(piece)

                token_count = self._token_count(plan.text)
                is_partial = any(r.is_partial for r in plan.blocks)
                registry_rows.append(
                    (
                        note_id,
                        fp,
                        chunk_index,
                        byte_start,
                        byte_end,
                        token_count,
                        "partial" if is_partial else "block",
                        None,  # parent_chunk_index
                        None,  # pdf_page_start
                        None,  # pdf_page_end
                    )
                )
                for ordinal, ref in enumerate(plan.blocks):
                    block_rows.append(
                        (note_id, chunk_index, ref.block_id, ordinal, 1 if ref.is_partial else 0)
                    )
                documents.append(
                    Document(
                        page_content=plan.text,
                        metadata={
                            "note_id": note_id,
                            "chunk_index": chunk_index,
                            "page_id": page_id,
                            "chunk_fingerprint": fp,
                            "source_block_ids": [r.block_id for r in plan.blocks],
                            "token_count": token_count,
                        },
                    )
                )
                chunk_index += 1

        annotated_md = "".join(parts)
        if self.generate_artifacts:
            chunked_path = CerebrumPaths().chunked_note_path(
                bubble_id=bubble_id, note_id=note_id
            )
            chunked_path.parent.mkdir(parents=True, exist_ok=True)
            chunked_path = chunked_path.parent / f"{note_id}.md"
            chunked_path.write_text(annotated_md, encoding="utf-8")

        self.note_chunk_registry.register_chunks(registry_rows)
        self.note_chunk_registry.register_chunk_blocks(note_id, block_rows)
        logging.info(
            f"Note: {note_id} block-chunked ({chunk_index} chunks across "
            f"{len(pages)} page(s)) + registered"
        )
        return annotated_md, documents
