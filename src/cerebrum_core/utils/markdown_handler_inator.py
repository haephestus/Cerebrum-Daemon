import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

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

        Returns:
            tuple[Path, FileMetadata]: Path to markdown file and enriched metadata
        """
        # Merge PDF metadata with any provided metadata
        combined_metadata = {**self.pdf_metadata, **(metadata or {})}

        # Use LLM to sanitize filename and generate metadata
        sanitized_metadata = self.sanitize_inator(
            filename=self.filepath.name, metadata=combined_metadata
        )

        # Clean the title to remove filesystem-unsafe characters
        sanitized_metadata.title = self._sanitize_filename(sanitized_metadata.title)

        domain = sanitized_metadata.domain
        subject = sanitized_metadata.subject
        filename = sanitized_metadata.title

        # Setup output path
        path = CerebrumPaths()
        markdown_dir = path.kb_artifacts_path() / domain / subject
        markdown_dir.mkdir(parents=True, exist_ok=True)

        # Convert PDF to markdown
        md_body = pymupdf4llm.to_markdown(self.filepath, show_progress=True)

        # Add YAML frontmatter
        yaml_front = self._yaml_inator(sanitized_metadata)
        full_md = f"{yaml_front}{md_body}"

        # Write to file
        md_output = markdown_dir / f"{filename}.md"
        md_output.write_text(full_md, encoding="utf-8")

        logger.info(f"Converted {self.filepath.name} → {md_output}")
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
    ):
        """
        Split markdown by headers and token limits, annotate with HTML comments.

        Args:
            markdown_path: Path to .md file with YAML frontmatter
            file_fingerprint: Unique identifier for the document

        Returns:
            Path to .chunked.md file with HTML comment annotations
        """
        if file_fingerprint:
            source_id = file_fingerprint
        else:
            source_id = note_id
        max_chunk_tokens = 4000

        # Extract YAML frontmatter
        yaml_pattern = re.compile(r"^(---\n.*?\n---\n\n)", re.S)
        yaml_match = yaml_pattern.match(markdown_text)

        if yaml_match:
            yaml_frontmatter = yaml_match.group(1)
            text = markdown_text[len(yaml_frontmatter) :]  # Content after YAML
        else:
            yaml_frontmatter = ""
            text = markdown_text

        # Split by markdown headers
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

        # Recursive splitter for oversized chunks
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens,
            chunk_overlap=200,
            length_function=lambda t: len(self.tokenizer.encode(t)),
            add_start_index=True,
        )

        # Process chunks
        processed_chunks = []
        for idx, chunk in enumerate(header_chunks):
            token_count = self._token_count(chunk.page_content)

            if token_count <= max_chunk_tokens:
                processed_chunks.append(chunk)
            else:
                # Split oversized chunks recursively
                sub_chunks = recursive_splitter.split_documents([chunk])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["parent_chunk_index"] = idx
                    # Preserve header metadata from parent chunk
                    for key, value in chunk.metadata.items():
                        if key not in sub_chunk.metadata:
                            sub_chunk.metadata[key] = value
                    processed_chunks.append(sub_chunk)

        # Build annotated markdown with byte coordinates
        output_lines = []
        registry_rows = []
        byte_cursor = 0

        for chunk_idx, chunk in enumerate(processed_chunks):
            content = chunk.page_content
            chunk_fingerprint = self._chunk_fingerprint(content)
            content_bytes = content.encode("utf-8")
            byte_length = len(content_bytes)

            parent_idx = chunk.metadata.get("parent_chunk_index", None)
            chunk_type = "recursive" if parent_idx is not None else "header"
            token_count = self._token_count(content)

            # Build HTML comment metadata block with header information
            metadata_lines = [
                "<!-- CHUNK_START",
                f"chunk_fingerprint: {chunk_fingerprint}",
                f"chunk_type: {chunk_type}",
                f"chunk_index: {chunk_idx}",
                f"parent_chunk_index: {parent_idx}",
                f"byte_start: {byte_cursor}",
                f"byte_end: {byte_cursor + byte_length}",
                f"token_count: {token_count}",
            ]

            # Add header hierarchy metadata
            for key, value in chunk.metadata.items():
                if key.startswith("Header") and value:
                    metadata_lines.append(f"{key.lower().replace(' ', '_')}: {value}")

            metadata_lines.append("-->")
            metadata_block = "\n".join(metadata_lines)

            output_lines.append(metadata_block)
            output_lines.append(content)
            output_lines.append("<!-- CHUNK_END -->")
            output_lines.append("")  # Blank line separator

            # Register chunk in database
            registry_rows.append(
                (
                    source_id,
                    chunk_fingerprint,
                    chunk_idx,
                    byte_cursor,
                    byte_cursor + byte_length,
                    token_count,
                    chunk_type,
                    parent_idx,
                )
            )

            byte_cursor += byte_length

        # Write chunked markdown (same directory as original .md)
        # Include YAML frontmatter at the top

        final_output = yaml_frontmatter + "\n".join(output_lines)

        # Register all chunks in database
        logger.info(f"Chunked {len(processed_chunks)} chunks → (in-memory)")

        return final_output, registry_rows, processed_chunks

    def _chunk_fingerprint(self, content):
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
