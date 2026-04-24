import logging

from langchain_core.documents.base import Document

from cerebrum_core.model_inator import NoteContent, NoteStorage
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.markdown_handler_inator import MarkdownChunker
from cerebrum_core.utils.registry.note_chunk_registry_inator import (
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


# modify to allow for chunking and chunk by chunk analysis
class NoteToMarkdownInator:
    """
    Converts an AppFlowy-style note into markdown
    """

    def __init__(self, convert_tables: bool = True) -> None:
        self.convert_tables = convert_tables

    # ------------ Core Public Method ------------- #
    def flatten(self, note: NoteContent) -> str:
        """
        Main entry point - returns a flattened Markdown string.
        """
        children = note.document["children"]
        lines = []

        for block in children:
            handler = getattr(self, f"_handle_{block['type'].replace('/','_')}", None)
            if handler:
                result = handler(block)
                if result:
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
    Chunks notes converted to markdown and registers chunks for analysis
    """

    def __init__(self, generate_artifacts: bool = True):
        super().__init__()
        self.note_chunk_registry = NoteChunkRegisterInator()
        self.generate_artifacts = generate_artifacts

    def chunk(
        self, flattened_note: str, note_id: str, bubble_id: str
    ) -> tuple[str, list[Document]]:
        annotated_md, registry_rows, documents = self.chunk_markdown(
            flattened_note, note_id=note_id
        )

        if self.generate_artifacts:
            chunked_path = CerebrumPaths().note_cache_path(bubble_id=bubble_id)
            chunked_path.write_text(annotated_md, encoding="utf-8")
            logging.info(f"Note: {note_id} chunked successfully")

        self.note_chunk_registry.register_chunks(registry_rows)
        logging.info(f"Note: {note_id} registerd successfully")
        return annotated_md, documents
