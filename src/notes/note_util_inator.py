import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.documents.base import Document

from common.file_util_inator import CerebrumPaths
from database.note_chunk_registry_inator import NoteChunkRegisterInator
from models.model_inator import Note, NoteManifest, Page, PageHistory, PageManifest
from notes.block_chunker_inator import Block, pack_blocks
from notes.markdown_handler_inator import MarkdownChunker


# convert the notes to markdown
# chunk notes and register chunks
def note_processor_inator(bubble_id: str, note_id: str, note: Note):
    # Block-aligned chunking (gap 1 / stream A): packs whole blocks, registers
    # note_chunks + the chunk↔block join. Replaces the old overlap-based path.
    NoteChunkerInator().chunk_note(note=note, note_id=note_id, bubble_id=bubble_id)
    logging.info(f"Note: {note_id} processed successfully")


def note_pages(note: Note) -> list[Page]:
    """A note's pages in display order (by page_index). Pages ARE the note's
    truth now, so this is just an ordered view used by chunking, analysis and
    sync so they all iterate pages consistently."""
    return sorted(note.pages, key=lambda p: p.page_index)


def diff_collapser_inator(note: Note) -> Note:
    """Collapse each page's content history so only the latest diff per version
    survives — prevents the per-page history.json from ballooning."""
    for page in note.pages:
        history = page.history.content
        if len(history) <= 1:
            continue
        latest_by_version: dict = {}
        version_order: list = []
        for entry in history:
            if entry.version not in latest_by_version:
                version_order.append(entry.version)
            latest_by_version[entry.version] = entry
        page.history.content = [latest_by_version[v] for v in version_order]
    return note


# ------------------------------ NOTE PACKAGING ------------------------------ #
#
# A note is a folder of pages. Pages are disk-truth; everything else derives
# from them.
#
#   notes/<note_id>/manifest.json          note-level: ids, title, analyse_note,
#                                          note version/hash, version_vector,
#                                          overview, and page_order — a
#                                          {page_id: index} hashtable that is the
#                                          single source of display order.
#   notes/<note_id>/pages/<page_id>/       folder name IS the stable page_id;
#                                          NEVER renamed (order lives in the note
#                                          manifest, not the folder name — so
#                                          reorder/delete don't move anything).
#       manifest.json                      page: id, index, content/ink hash+
#                                          version, version_vector, last_modified.
#       content.json                       just `{"document": {...}}`.
#       ink.json                           just `{"ink": [...]}`.
#       history.json                       `{"content": [...], "ink": [...]}`.
#       analysis.json                      per-page analysis SIDECAR — owned by
#                                          write/read_page_analysis; _save_note
#                                          never touches it (so a plain save can't
#                                          wipe analysis, and delete-page drops it
#                                          with the folder).
#
# All plain, human-readable JSON. `filename` everywhere else (URL params,
# NoteOut.filename, note_registry) is UNCHANGED — still `<note_id>.json`; it only
# derives the folder name (stripping ".json").
#
# Legacy self-migration: `_load_note` reads the current per-page-manifest layout,
# then falls back to the previous single-manifest+content layout, the earlier
# single content.json, and finally a legacy flat file — mapping each into a Note.
# `_save_note` always writes the current layout and retires the old files, so any
# note self-migrates the next time it is saved.
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
    half-written JSON file behind.

    The temp file gets a UNIQUE name (mkstemp) in the target's directory, not a
    shared `<name>.tmp`. Two writers of the same file — e.g. a client `_save_note`
    and the analysis pipeline's `write_note_overview` racing on one note's
    manifest.json — otherwise both truncate+write the SAME `.tmp`, interleaving
    into a file that is a complete JSON object followed by the longer payload's
    leftover tail: the `json.JSONDecodeError: Extra data` that bricked bubble
    loads. A per-writer temp + atomic `os.replace` makes concurrent writes safe
    (last replace wins, whole-file, never corrupt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())  # durability: content is on disk before replace
        os.replace(tmp_name, path)
    except BaseException:
        # Don't leave the unique temp behind if the write/replace failed.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _safe_read_json(path: Path, default):
    """Read a JSON file, returning `default` if it's missing OR corrupt. Used for
    the page SIDECARS (ink/history) so one garbled sidecar — e.g. a note left with
    a malformed history.json by an older buggy writer — degrades that field rather
    than bricking the whole note load. The next _save_note rewrites it cleanly."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logging.warning("Ignoring corrupt sidecar %s — using default", path)
        return default


def _ordered_page_ids(page_order: Any, pages_root: Path) -> list[str]:
    """Resolve the display-ordered page_ids from a manifest's page_order —
    accepting the current {page_id: index} hashtable OR a legacy [page_id, ...]
    list. Falls back to whatever folders exist on disk (sorted) if page_order is
    missing/empty."""
    if isinstance(page_order, dict) and page_order:
        return [pid for pid, _ in sorted(page_order.items(), key=lambda kv: kv[1])]
    if isinstance(page_order, list) and page_order:
        return list(page_order)
    if pages_root.exists():
        return sorted(d.name for d in pages_root.iterdir() if d.is_dir())
    return []


def _load_page_folder(pages_root: Path, pid: str, order_index: int) -> Optional[Page]:
    """Read one `pages/<pid>/` folder into a Page. Handles the current layout
    (manifest.json + pure content.json) and the previous one (page metadata
    embedded in content.json). Returns None if the page has no content.json."""
    pdir = pages_root / pid
    cpath = pdir / "content.json"
    if not cpath.exists():
        return None
    # Degrade a corrupt content/manifest to defaults rather than crashing the
    # whole note (and the bubble listing) — same resilience as the note manifest;
    # the concurrent-write bug in _atomic_write_text could corrupt any of these.
    cdata = _safe_read_json(cpath, {})
    if not isinstance(cdata, dict):
        cdata = {}
    document = cdata.get("document", {})

    mpath = pdir / "manifest.json"
    if mpath.exists():
        meta = _safe_read_json(mpath, {})
        if not isinstance(meta, dict):
            meta = {}
    else:  # previous layout embedded the page metadata inside content.json
        meta = dict(cdata.get("metadata", {}))
        meta.setdefault("page_id", cdata.get("page_id", pid))
        meta.setdefault("page_index", cdata.get("page_index", order_index))
    meta.setdefault("page_id", pid)
    meta.setdefault("page_index", order_index)

    ink = _safe_read_json(pdir / "ink.json", {}).get("ink", [])
    hist = _safe_read_json(pdir / "history.json", {})

    return Page(
        page_id=pid,
        page_index=meta.get("page_index", order_index),
        document=document,
        ink=ink,
        history=PageHistory(**hist) if hist else PageHistory(),
        metadata=PageManifest(**meta),
    )


def _note_manifest_from_raw(raw: dict, pages: list[Page]) -> NoteManifest:
    """Build a NoteManifest from a note manifest.json — current (flat fields) or
    legacy (note version fields nested under a `metadata` key). page_order is
    always rebuilt from the loaded pages."""
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

    def pick(key, default):
        if key in raw:
            return raw[key]
        return meta.get(key, default)

    return NoteManifest(
        title=raw.get("title", ""),
        note_id=raw.get("note_id", ""),
        bubble_id=raw.get("bubble_id", ""),
        analyse_note=raw.get("analyse_note", True),
        content_hash=pick("content_hash", ""),
        content_version=pick("content_version", 0),
        ink_hash=pick("ink_hash", ""),
        ink_version=pick("ink_version", 0),
        version_vector=pick("version_vector", {}) or {},
        last_modified=pick("last_modified", None) or datetime.now(),
        overview=raw.get("overview"),
        page_order={p.page_id: p.page_index for p in pages},
    )


def _note_from_legacy_flat(data: dict) -> Note:
    """Map an old single-blob NoteStorage dict (flat file or previous single
    content.json) into a Note. Uses its `pages` if present, else synthesises one
    page ('p1') from its content/ink; old note-level history moves onto page 1."""
    page_dicts = data.get("pages") or []
    pages: list[Page] = []
    if page_dicts:
        for i, pd in enumerate(page_dicts):
            meta = dict(pd.get("metadata", {}))
            meta.setdefault("page_id", pd.get("page_id", f"p{i + 1}"))
            meta.setdefault("page_index", pd.get("page_index", i))
            hist = pd.get("history") or {}
            pages.append(
                Page(
                    page_id=pd.get("page_id", f"p{i + 1}"),
                    page_index=pd.get("page_index", i),
                    document=pd.get("document", {}),
                    ink=pd.get("ink", []),
                    history=PageHistory(**hist) if hist else PageHistory(),
                    metadata=PageManifest(**meta),
                )
            )
    else:
        content = data.get("content") or {}
        document = content.get("document", {}) if isinstance(content, dict) else {}
        meta = dict(data.get("metadata", {}))
        meta.setdefault("page_id", "p1")
        meta.setdefault("page_index", 0)
        hist = data.get("history") or {}
        pages.append(
            Page(
                page_id="p1",
                page_index=0,
                document=document,
                ink=data.get("ink", []),
                history=PageHistory(**hist) if hist else PageHistory(),
                metadata=PageManifest(**meta),
            )
        )
    return Note(manifest=_note_manifest_from_raw(data, pages), pages=pages)


def _load_note(notes_dir: Path, filename: str, *, skip_ink: bool = False) -> Note:
    """Load a note as a Note. Reads the current per-page-manifest folder layout;
    falls back to the previous single content.json/ink.json, then a legacy flat
    file — each mapped into a Note so old notes self-migrate on next save. With
    skip_ink, page ink is left empty (for listing)."""
    note_dir = _note_dir(notes_dir, filename)
    manifest_path = note_dir / "manifest.json"

    if manifest_path.exists():
        # A corrupt note manifest must NOT brick the whole note (and, via the
        # bubble listing, every other note in the bubble). Degrade to an empty
        # manifest: page content is the source of truth and lives in the page
        # folders — `_ordered_page_ids` falls back to the on-disk folders when
        # page_order is missing, so the note still loads. Title/overview are lost
        # until the next save rewrites a clean manifest. See _atomic_write_text
        # for the concurrent-write bug that produced such files.
        raw = _safe_read_json(manifest_path, {})
        if not isinstance(raw, dict):
            raw = {}
        pages_root = note_dir / "pages"
        pages: list[Page] = []
        for idx, pid in enumerate(_ordered_page_ids(raw.get("page_order"), pages_root)):
            page = _load_page_folder(pages_root, pid, idx)
            if page is None:
                continue
            if skip_ink:
                page.ink = []
            pages.append(page)
        pages.sort(key=lambda p: p.page_index)
        return Note(manifest=_note_manifest_from_raw(raw, pages), pages=pages)

    content_path = note_dir / "content.json"
    if content_path.exists():  # previous single-file layout (may carry `pages`)
        stored_data = json.loads(content_path.read_text(encoding="utf-8"))
        if not skip_ink:
            ink_path = note_dir / "ink.json"
            stored_data["ink"] = (
                json.loads(ink_path.read_text(encoding="utf-8")).get("ink", [])
                if ink_path.exists()
                else stored_data.get("ink", [])
            )
        return _note_from_legacy_flat(stored_data)

    # Legacy flat file.
    legacy_path = _legacy_flat_filepath(notes_dir, filename)
    stored_data = json.loads(legacy_path.read_text(encoding="utf-8"))
    return _note_from_legacy_flat(stored_data)


def _load_note_skip_ink(notes_dir: Path, filename: str) -> Note:
    """For listing: the note + page documents but no ink loaded."""
    return _load_note(notes_dir, filename, skip_ink=True)


def _note_needs_ink_migration(notes_dir: Path, filename: str) -> bool:
    """True if the first page has no ink.json yet (older layouts stored ink
    elsewhere) — signals _save_note to write ink even if it looks unchanged."""
    note_dir = _note_dir(notes_dir, filename)
    manifest_path = note_dir / "manifest.json"
    if manifest_path.exists():
        pages_root = note_dir / "pages"
        ordered = _ordered_page_ids(
            json.loads(manifest_path.read_text(encoding="utf-8")).get("page_order"),
            pages_root,
        )
        if not ordered:
            return True
        return not (pages_root / ordered[0] / "ink.json").exists()
    return not (note_dir / "ink.json").exists()


def _save_note(
    notes_dir: Path,
    filename: str,
    note: Note,
    *,
    write_ink: bool = True,
) -> None:
    """
    Write the per-page folder layout:

        notes/<id>/manifest.json            note-level manifest incl. page_order
        notes/<id>/pages/<page_id>/
            manifest.json                   page id/index/versions/vector
            content.json                    {"document": {...}}
            ink.json                        {"ink": [...]}  (skipped if write_ink=False)
            history.json                    page diff journals

    analysis.json is a sidecar and is NEVER touched here. Page folders whose
    page_id is no longer present are removed (and their analysis.json goes with
    them); the previous content.json/ink.json and any legacy flat file are retired.
    """
    note_dir = _note_dir(notes_dir, filename)
    pages_root = note_dir / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)

    pages = note_pages(note)

    # page_order is the authoritative display order — rebuilt from the pages.
    note.manifest.page_order = {p.page_id: p.page_index for p in pages}
    manifest = json.loads(note.manifest.model_dump_json())
    # Preserve the note-level analysis overview (written out-of-band by
    # write_note_overview) if this Note in memory doesn't carry it — a plain
    # save must not wipe it.
    manifest_path = note_dir / "manifest.json"
    if manifest.get("overview") is None and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("overview"):
                manifest["overview"] = old["overview"]
        except Exception:
            pass
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, default=str))

    keep: set[str] = set()
    for p in pages:
        pdir = pages_root / p.page_id
        pdir.mkdir(parents=True, exist_ok=True)
        keep.add(p.page_id)
        # Keep the page manifest's own id/index in lockstep with the Page.
        p.metadata.page_id = p.page_id
        p.metadata.page_index = p.page_index
        _atomic_write_text(pdir / "manifest.json", p.metadata.model_dump_json(indent=2))
        _atomic_write_text(
            pdir / "content.json", json.dumps({"document": p.document}, indent=2)
        )
        if write_ink:
            _atomic_write_text(pdir / "ink.json", json.dumps({"ink": p.ink}, indent=2))
        _atomic_write_text(pdir / "history.json", p.history.model_dump_json(indent=2))

    # Drop page folders (incl. their analysis.json) for pages that were removed.
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


def write_page_analysis(
    notes_dir: Path, filename: str, page_analyses: Dict[str, Any]
) -> None:
    """Write per-page analysis.json into each page folder (page-aware analysis).
    `page_analyses` maps page_id → that page's analysis dict. Pages not present
    on disk are skipped (a chunk whose page was since deleted is just dropped)."""
    pages_root = _note_dir(notes_dir, filename) / "pages"
    for page_id, analysis in page_analyses.items():
        pdir = pages_root / page_id
        if not pdir.exists():
            continue
        _atomic_write_text(
            pdir / "analysis.json", json.dumps(analysis, indent=2, default=str)
        )


def read_page_analysis(notes_dir: Path, filename: str, page_id: str) -> Optional[dict]:
    """A page's cached analysis.json, or None."""
    apath = _note_dir(notes_dir, filename) / "pages" / page_id / "analysis.json"
    if apath.exists():
        return json.loads(apath.read_text(encoding="utf-8"))
    return None


def write_note_overview(notes_dir: Path, filename: str, overview: dict) -> None:
    """Merge the note-level analysis overview into manifest.json (general note
    data lives there). Read-modify-write so the rest of the manifest is kept."""
    mpath = _note_dir(notes_dir, filename) / "manifest.json"
    if not mpath.exists():
        return
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return
    manifest["overview"] = overview
    _atomic_write_text(mpath, json.dumps(manifest, indent=2, default=str))


def read_note_overview(notes_dir: Path, filename: str) -> Optional[dict]:
    """The note-level overview from the manifest, or None."""
    mpath = _note_dir(notes_dir, filename) / "manifest.json"
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8")).get("overview")
        except Exception:
            return None
    return None


# modify to allow for chunking and chunk by chunk analysis
class NoteToMarkdownInator:
    """
    Converts an AppFlowy-style note into markdown
    """

    def __init__(self, convert_tables: bool = True) -> None:
        self.convert_tables = convert_tables

    # ------------ Core Public Method ------------- #
    def flatten(self, document: Dict[str, Any]) -> str:
        children = document["children"]
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
    def flatten_blocks(self, document: Dict[str, Any]) -> list[Block]:
        """Per-block (id, type, rendered markdown) — the structured parallel to
        flatten(), used by block-aligned chunking.

        Block id resolution, best → worst, for DURABLE tap-a-block mapping:
          1. client-persisted id (``block["id"]``, the AppFlowy node id) — stable
             across edits/reorders/sync, mirrors how ink strokes carry ids.
          2. content-addressed fallback (``blk_<hash(type+text)>``) when the
             client didn't send an id — survives reorder (id follows content, not
             slot) and only changes when the block's content changes, which is
             exactly when its old analysis goes stale. An occurrence counter
             disambiguates identical blocks on the same page.
          3. positional (``blk{i}``) only as a last-resort tiebreaker, never as
             the identity on its own — index-fragile, so we never anchor on it.

        Blocks that render empty are dropped."""
        out: list[Block] = []
        seen_hashes: dict[str, int] = {}  # content hash → occurrences so far
        missing_id_count = 0
        for i, block in enumerate((document or {}).get("children", [])):
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

            client_id = block.get("id")
            if client_id:
                block_id = client_id
            else:
                missing_id_count += 1
                digest = hashlib.sha1(
                    f"{btype}\x00{result}".encode("utf-8")
                ).hexdigest()[:12]
                occ = seen_hashes.get(digest, 0)
                seen_hashes[digest] = occ + 1
                # occurrence counter + positional index disambiguate duplicate
                # blocks; the hash still anchors identity to the content.
                block_id = f"blk_{digest}_{occ}" if occ else f"blk_{digest}"

            out.append(Block(block_id=block_id, block_type=btype, text=result))

        if missing_id_count:
            logging.warning(
                "[flatten_blocks] %d/%d block(s) arrived without a client id — "
                "using content-hash fallback ids. Tap-a-block is only fully "
                "durable once the client persists AppFlowy node ids.",
                missing_id_count,
                len(out),
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
        """(page_id, document) per page — from a Note's pages, or a single
        synthesised page for a bare document dict."""
        if isinstance(note, Note):
            return [(p.page_id, p.document) for p in note_pages(note)]
        if isinstance(note, dict):
            return [("p0", note)]
        return [("p0", getattr(note, "document", {}) or {})]

    def chunk_note(
        self, note, note_id: str, bubble_id: str
    ) -> tuple[str, list[Document]]:
        """Block-aligned chunking (gap 1). Chunks EACH PAGE separately so a chunk
        never crosses a page boundary (page > chunk > block), with a running
        global chunk_index and byte offsets into one assembled .md (byte-offset
        addressable exactly like before, so the analyser's chunk_fetcher keeps
        working). Registers `note_chunks` + the `note_chunk_blocks` join.

        `note` may be a Note (multi-page) or a bare document dict (single doc);
        `_pages_for_chunking` yields one synthesised page for the bare shape.
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
            blocks = md.flatten_blocks(document or {})
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
                        page_id,
                    )
                )
                for ordinal, ref in enumerate(plan.blocks):
                    block_rows.append(
                        (
                            note_id,
                            chunk_index,
                            ref.block_id,
                            ordinal,
                            1 if ref.is_partial else 0,
                        )
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
