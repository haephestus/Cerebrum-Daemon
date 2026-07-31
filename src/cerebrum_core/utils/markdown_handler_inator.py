import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, cast

import pymupdf4llm
import tiktoken
import yaml
from langchain_ollama import OllamaLLM
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agents.rose import RosePrompts
from cerebrum_core.constants import DEFAULT_CHAT_MODEL
from cerebrum_core.model_inator import FileMetadata
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths

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


class MarkdownConverter:
    """
    Converts PDF files to Markdown with LLM-enriched YAML frontmatter.
    Handles file sanitization and metadata generation.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.fingerprint = self._fingerprint_inator(filepath)
        self.pdf_metadata = self._extract_pdf_metadata(filepath)

    def convert(self, metadata: Optional[dict]) -> tuple[Path, FileMetadata]:
        """
        Convert PDF to markdown with LLM-sanitized metadata.
        Also emits a sidecar .pageoffsets.json mapping markdown byte ranges to PDF page numbers.
        """
        combined_metadata = {**self.pdf_metadata, **(metadata or {})}
        sanitized_metadata = self.sanitize_inator(
            filename=self.filepath.name, metadata=combined_metadata
        )
        sanitized_metadata.title = self._sanitize_filename(sanitized_metadata.title)

        domain = sanitized_metadata.domain
        subject = sanitized_metadata.subject
        filename = sanitized_metadata.title

        path = CerebrumPaths()
        markdown_dir = path.kb_artifacts_path(
            domain=domain, subject=subject, sanitised_name=filename
        )
        markdown_dir.mkdir(parents=True, exist_ok=True)

        # --- CHANGED: page_chunks=True instead of flat string ---
        pages = cast(
            list[dict],
            pymupdf4llm.to_markdown(
                self.filepath, page_chunks=True, show_progress=True
            ),
        )

        body_parts = []
        page_offsets = []  # [(byte_start, byte_end, pdf_page_num), ...]
        cursor = 0
        for page in pages:
            text = page["text"]
            text_bytes = len(text.encode("utf-8"))
            page_num = page["metadata"]["page"]  # confirmed 1-indexed from test run
            page_offsets.append((cursor, cursor + text_bytes, page_num))
            body_parts.append(text)
            cursor += text_bytes

        md_body = "".join(body_parts)

        yaml_front = self._yaml_inator(sanitized_metadata)
        full_md = f"{yaml_front}{md_body}"

        md_output = markdown_dir / f"{filename}.md"
        md_output.write_text(full_md, encoding="utf-8")

        # --- ADDED: sidecar page-offset index ---#
        offsets_path = markdown_dir / f"{filename}.pageoffsets.json"
        offsets_path.write_text(json.dumps(page_offsets), encoding="utf-8")

        logger.info(f"Converted {self.filepath.name} → {md_output}")
        logger.info(f"Wrote page offsets ({len(page_offsets)} pages) → {offsets_path}")
        return md_output, sanitized_metadata

    def sanitize_inator(self, filename: str, metadata: dict | None) -> FileMetadata:
        """
        Use LLM to sanitize filename and enrich metadata.
        Offloading renaming and sanitization to LLM for consistent categorization.
        """
        chat_model = (
            ConfigManager().load_config().models.chat_model or DEFAULT_CHAT_MODEL
        )

        metadata_json = json.dumps(metadata, indent=2) if metadata else "{}"
        sanitize_prompt = RosePrompts.get_prompt("rose_rename")

        if not sanitize_prompt:
            raise ValueError("Prompt 'rose_rename' not found in RosePrompts")

        filled_prompt = sanitize_prompt.format(
            filename=filename, metadata=metadata_json
        )

        sanitized_response = OllamaLLM(model=chat_model).invoke(filled_prompt)
        logger.info(f"LLM sanitization response: {sanitized_response}")

        try:
            parsed_response = json.loads(sanitized_response)
        except json.JSONDecodeError:

            match = re.search(
                r"```json\s*(\{.*?\})\s*```", sanitized_response, re.DOTALL
            )
            if match:
                try:
                    parsed_response = json.loads(match.group(1))
                except json.JSONDecodeError:
                    raise ValueError(
                        f"LLM did not return valid JSON: {sanitized_response}"
                    )
            else:
                raise ValueError(f"LLM did not return valid JSON: {sanitized_response}")
        return FileMetadata(**parsed_response)

    def _fingerprint_inator(self, filepath: Path) -> str:
        """Generate unique fingerprint for document based on content."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _yaml_inator(self, metadata: FileMetadata) -> str:
        """Generate YAML frontmatter from metadata."""
        yaml_dump = yaml.dump(metadata.model_dump(), sort_keys=False)
        return f"---\n{yaml_dump}---\n\n"

    def _sanitize_filename(self, filename: str) -> str:
        """
        Remove or replace filesystem-unsafe characters from filename.
        Preserves hyphens and underscores for readability.
        """
        # Replace common problematic characters
        replacements = {
            "/": "-",
            "\\": "-",
            ":": "-",
            "*": "",
            "?": "",
            '"': "",
            "<": "",
            ">": "",
            "|": "-",
        }

        sanitized = filename
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        # Remove multiple consecutive hyphens
        while "--" in sanitized:
            sanitized = sanitized.replace("--", "-")

        # Remove leading/trailing hyphens
        sanitized = sanitized.strip("-")

        return sanitized

    def _extract_pdf_metadata(self, filepath: Path) -> dict:
        """Extract metadata from PDF file using PyMuPDF."""
        import pymupdf

        try:
            doc = pymupdf.open(filepath)
            metadata = doc.metadata
            doc.close()

            # Clean up metadata - remove None values and empty strings
            cleaned_metadata = {}

            if metadata:
                if metadata.get("author"):
                    # Split multiple authors if separated by common delimiters
                    authors = metadata["author"]
                    if ";" in authors:
                        cleaned_metadata["authors"] = [
                            a.strip() for a in authors.split(";")
                        ]
                    elif "," in authors and " and " not in authors.lower():
                        cleaned_metadata["authors"] = [
                            a.strip() for a in authors.split(",")
                        ]
                    else:
                        cleaned_metadata["authors"] = [authors.strip()]

                if metadata.get("title"):
                    cleaned_metadata["title"] = metadata["title"].strip()

                if metadata.get("subject"):
                    cleaned_metadata["subject"] = metadata["subject"].strip()

                if metadata.get("keywords"):
                    # Keywords might be comma-separated
                    keywords = metadata["keywords"]
                    if "," in keywords:
                        cleaned_metadata["keywords"] = [
                            k.strip() for k in keywords.split(",")
                        ]
                    else:
                        cleaned_metadata["keywords"] = [keywords.strip()]

                # Additional metadata that might be useful
                if metadata.get("creator"):
                    cleaned_metadata["creator"] = metadata["creator"].strip()

                if metadata.get("producer"):
                    cleaned_metadata["producer"] = metadata["producer"].strip()

            logger.info(f"Extracted PDF metadata: {cleaned_metadata}")
            return cleaned_metadata

        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            return {}


class MarkdownChunker:
    """
    Splits markdown into semantic chunks with byte-coordinate tracking.
    Generates .chunked.md files with HTML comment annotations.

    Args:
        use_file_registry:  toggles between file_registry or note_registry
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_markdown(
        self,
        markdown_text: str,
        *,
        file_fingerprint: Optional[str] = None,
        note_id: Optional[str] = None,
        page_offsets: Optional[list[tuple[int, int, int]]] = None,
    ):
        if file_fingerprint:
            source_id = file_fingerprint
        else:
            source_id = note_id
        max_chunk_tokens = 512

        yaml_pattern = re.compile(r"^(---\n.*?\n---\n\n)", re.S)
        yaml_match = yaml_pattern.match(markdown_text)

        if yaml_match:
            yaml_frontmatter = yaml_match.group(1)
            text = markdown_text[len(yaml_frontmatter) :]
        else:
            yaml_frontmatter = ""
            text = markdown_text

        yaml_frontmatter_bytes = len(yaml_frontmatter.encode("utf-8"))

        header_levels = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=header_levels, strip_headers=False
        )
        header_chunks = header_splitter.split_text(text)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens,
            chunk_overlap=200,
            length_function=lambda t: len(self.tokenizer.encode(t)),
            add_start_index=True,
        )

        header_processed_chunks = []
        for idx, chunk in enumerate(header_chunks):
            token_count = self._token_count(chunk.page_content)
            if token_count <= max_chunk_tokens:
                header_processed_chunks.append(chunk)
            else:
                sub_chunks = recursive_splitter.split_documents([chunk])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["parent_chunk_index"] = idx
                    for key, value in chunk.metadata.items():
                        if key not in sub_chunk.metadata:
                            sub_chunk.metadata[key] = value
                    header_processed_chunks.append(sub_chunk)

        output_lines = []
        registry_rows = []
        processed_chunks = []
        final_output = yaml_frontmatter
        byte_cursor = 0  # relative to `text` — unaffected by frontmatter, used only for layout math

        start_tag_pattern = re.compile(r"<!--\s*CHUNK_START.*?-->", re.DOTALL)
        end_tag_pattern = re.compile(r"<!--\s*CHUNK_END\s*-->")
        BYTE_FIELD_WIDTH = 10

        for chunk_idx, chunk in enumerate(header_processed_chunks):
            raw_content = chunk.page_content
            clean_content = start_tag_pattern.sub("", raw_content)
            clean_content = end_tag_pattern.sub("", clean_content).strip("\n")
            chunk.page_content = clean_content

            chunk_fingerprint = self._chunk_fingerprint(clean_content)
            content_bytes = clean_content.encode("utf-8")
            byte_length = len(content_bytes)
            block_ids = re.findall(r"<!-- block_id:(\S+) -->", clean_content)

            parent_idx = chunk.metadata.get("parent_chunk_index", None)
            chunk_type = "recursive" if parent_idx is not None else "header"
            token_count = self._token_count(clean_content)

            header_meta_lines = [
                "<!-- CHUNK_START",
                f"chunk_fingerprint: {chunk_fingerprint}",
                f"chunk_type: {chunk_type}",
                f"chunk_index: {chunk_idx}",
                f"source_block_ids: {json.dumps(block_ids)}",
                f"parent_chunk_index: {parent_idx}",
                f"token_count: {token_count}",
            ]
            for key, value in chunk.metadata.items():
                if key.startswith("Header") and value:
                    header_meta_lines.append(
                        f"{key.lower().replace(' ', '_')}: {value}"
                    )

            if page_offsets:
                page_start, page_end = self._resolve_page_range(
                    page_offsets,
                    byte_cursor,
                    byte_cursor + byte_length,
                )
            else:
                page_start = page_end = None

            metadata_lines = header_meta_lines + [
                "byte_start: {:0{w}d}".format(0, w=BYTE_FIELD_WIDTH),
                "byte_end: {:0{w}d}".format(0, w=BYTE_FIELD_WIDTH),
                f"pdf_page_start: {page_start}",
                f"pdf_page_end: {page_end}",
                "-->",
            ]
            metadata_block = "\n".join(metadata_lines)
            metadata_size = len(metadata_block.encode("utf-8")) + 1

            # The one separator byte between the PREVIOUS chunk's trailing
            # "" element and THIS chunk's metadata_block. output_lines is
            # only empty for chunk_idx == 0 — every chunk after that needs
            # this extra byte accounted for before metadata_block's own
            # position, or content_byte_start lands one byte early (on the
            # join separator itself, not the first real content byte).
            sep_before = 1 if output_lines else 0

            content_byte_start = byte_cursor + sep_before + metadata_size
            content_byte_end = content_byte_start + byte_length

            byte_start_actual = content_byte_start + yaml_frontmatter_bytes
            byte_end_actual = content_byte_end + yaml_frontmatter_bytes

            metadata_lines = header_meta_lines + [
                "byte_start: {:0{w}d}".format(byte_start_actual, w=BYTE_FIELD_WIDTH),
                "byte_end: {:0{w}d}".format(byte_end_actual, w=BYTE_FIELD_WIDTH),
                f"pdf_page_start: {page_start}",
                f"pdf_page_end: {page_end}",
                "-->",
            ]
            metadata_block = "\n".join(metadata_lines)

            chunk.metadata["chunk_index"] = chunk_idx
            chunk.metadata["source_block_ids"] = block_ids
            chunk.metadata["byte_start"] = byte_start_actual
            chunk.metadata["byte_end"] = byte_end_actual
            chunk.metadata["chunk_fingerprint"] = chunk_fingerprint
            chunk.metadata["chunk_type"] = chunk_type
            chunk.metadata["note_id"] = source_id
            chunk.metadata["pdf_page_start"] = page_start
            chunk.metadata["pdf_page_end"] = page_end

            # Incremental byte tracking, now using the SAME sep_before this
            # chunk's content_byte_start used above — the two must agree,
            # or the running cursor and the position math it feeds drift
            # apart again exactly like this bug.
            new_pieces = [metadata_block, clean_content, "<!-- CHUNK_END -->", ""]
            byte_cursor += sep_before
            for i, piece in enumerate(new_pieces):
                if i > 0:
                    byte_cursor += 1
                byte_cursor += len(piece.encode("utf-8"))
            output_lines.extend(new_pieces)

            registry_rows.append(
                (
                    source_id,
                    chunk_fingerprint,
                    chunk_idx,
                    byte_start_actual,
                    byte_end_actual,
                    token_count,
                    chunk_type,
                    parent_idx,
                    page_start,
                    page_end,
                )
            )
            processed_chunks.append(chunk)

        final_output = yaml_frontmatter + "\n".join(output_lines)
        logger.info(f"Chunked {len(processed_chunks)} chunks")

        return final_output, registry_rows, processed_chunks

    def _resolve_page_range(self, page_offsets, byte_start, byte_end):
        pages_in_range = [
            p for (s, e, p) in page_offsets if byte_start < e and byte_end > s
        ]
        if not pages_in_range:
            preceding = [p for (s, e, p) in page_offsets if s <= byte_start]
            page = preceding[-1] if preceding else page_offsets[0][2]
            return page, page
        return min(pages_in_range), max(pages_in_range)

    def _chunk_fingerprint(self, content):
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
