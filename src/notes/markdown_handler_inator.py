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
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agents.rose import RosePrompts
from cerebrum_core.constants import DEFAULT_CHAT_MODEL
from models.model_inator import FileMetadata
from cerebrum_core.user_inator import ConfigManager
from common.file_util_inator import CerebrumPaths

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

        # If the source is a scanned/text-layerless PDF, OCR it in place first
        # so everything downstream (to_markdown, TOC, page offsets) reads a
        # real text layer. Replaces the source PDF so re-chunks reuse it.
        self._maybe_ocr_in_place()

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

        # --- ADDED: sidecar TOC / structural outline ---#
        toc = self._extract_toc(self.filepath)
        toc_path = markdown_dir / f"{filename}.toc.json"
        toc_path.write_text(json.dumps(toc), encoding="utf-8")

        # --- PAUSED: global figure extraction --------------------------------
        # Disabled for now. Prior image extraction ballooned per-textbook
        # storage; the bbox + crop-on-demand approach (client renders the
        # region straight from the served PDF via /figure/{fp}/{idx}) removes
        # the need to store images at all. When re-enabled, gate this to exam
        # papers only — where figures are load-bearing (e.g. geometry
        # diagrams) — via sanitized_metadata.doc_type == "exam_paper". The
        # _extract_figures / _nearest_caption methods and the figures table +
        # serve routes remain in place, just unfed. TODO(review): re-enable
        # for exams.
        #
        # figures = self._extract_figures(self.filepath)
        # figures_path = markdown_dir / f"{filename}.figures.json"
        # figures_path.write_text(json.dumps(figures), encoding="utf-8")

        logger.info(f"Converted {self.filepath.name} → {md_output}")
        logger.info(f"Wrote page offsets ({len(page_offsets)} pages) → {offsets_path}")
        logger.info(f"Wrote TOC ({len(toc)} entries) → {toc_path}")
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

        # keep_alive=0: drop the chat model from Ollama's memory as soon as
        # this one metadata call returns. Otherwise Ollama keeps it resident
        # for its default 5 min, so it's still pinned when the embedding phase
        # loads its own model — two large models in RAM at once was a big part
        # of what pushed this box into HDD swap during a conversion.
        sanitized_response = OllamaLLM(model=chat_model, keep_alive=0).invoke(
            filled_prompt
        )
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

    def _is_scanned(
        self, filepath: Path, sample: int = 8, min_chars_per_page: int = 40
    ) -> bool:
        """
        Heuristic scanned-PDF detector: sample up to `sample` pages evenly and
        treat the document as scanned when the average extractable text per
        page falls below `min_chars_per_page`. Born-digital PDFs have a text
        layer and comfortably clear the bar; image-only scans yield ~nothing.
        """
        import pymupdf

        doc = pymupdf.open(filepath)
        n = len(doc)
        if n == 0:
            doc.close()
            return False

        if n <= sample:
            idxs = range(n)
        else:
            idxs = [round(i * (n - 1) / (sample - 1)) for i in range(sample)]

        total = 0
        counted = 0
        for i in idxs:
            total += len(doc[i].get_text("text").strip())
            counted += 1
        doc.close()

        avg = total / counted if counted else 0
        return avg < min_chars_per_page

    def _maybe_ocr_in_place(self) -> None:
        """
        If the source PDF looks scanned, run ocrmypdf to add a text layer and
        REPLACE the source file in place — the searchable PDF becomes
        canonical, so future re-chunks read real text and don't re-OCR (the
        scanned detector returns False on the OCR'd file). No-op for
        born-digital PDFs, and a logged no-op (never a crash) when
        ocrmypdf / Tesseract / Ghostscript aren't available.
        """
        try:
            if not self._is_scanned(self.filepath):
                return
        except Exception as e:
            logger.warning(f"Scanned-PDF detection failed ({e}); skipping OCR")
            return

        try:
            import ocrmypdf
        except ImportError:
            logger.warning(
                "ocrmypdf not installed — scanned PDF left as-is (text will be sparse)"
            )
            return

        logger.info(f"Scanned PDF detected — running OCR: {self.filepath.name}")
        tmp_out = self.filepath.with_suffix(".ocr.tmp.pdf")
        try:
            ocrmypdf.ocr(
                self.filepath,
                tmp_out,
                skip_text=True,  # leave any pages that already have a text layer
                optimize=0,  # avoid optional optimiser deps (pngquant/jbig2enc)
                progress_bar=False,
                language="eng",
                # Cap parallelism: ocrmypdf defaults to one worker per core (8
                # here), each running Ghostscript rasterisation + Tesseract on a
                # page — a multi-GB burst that shoved this box into HDD swap and
                # froze it. 2 keeps OCR usable without the memory spike.
                jobs=2,
            )
            os.replace(tmp_out, self.filepath)
            logger.info(f"OCR applied; source PDF replaced: {self.filepath.name}")
        except Exception as e:
            logger.warning(f"ocrmypdf failed ({e}); keeping original PDF")
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except OSError:
                    pass

    def _extract_toc(self, filepath: Path) -> list:
        """
        Extract the PDF's embedded outline (bookmarks) as a flat list of
        [level, title, page] with 1-indexed pages — the same page space the
        .pageoffsets sidecar uses, so chunk pages map straight onto it.

        Entries with no resolved page destination (page <= 0) are dropped.
        Returns [] when the PDF ships no embedded outline (common for
        scanned or exported PDFs — the printed-TOC fallback is a later step).
        """
        import pymupdf

        try:
            doc = pymupdf.open(filepath)
            raw = doc.get_toc(simple=True)  # [[level, title, page], ...]
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to extract TOC: {e}")
            return []

        toc = [
            [level, title, page]
            for (level, title, page) in raw
            if page and page > 0
        ]
        logger.info(f"Extracted TOC: {len(toc)} entries")
        return toc

    # Caption anchors near a figure, and the minimum size (points) below which
    # an image is treated as decoration (icon, rule, bullet) and skipped.
    _CAPTION_RE = re.compile(
        r"^\s*(fig(?:ure)?|table|diagram|graph|chart|plate|scheme)\b", re.I
    )
    _MIN_FIGURE_PT = 50.0

    def _extract_figures(self, filepath: Path) -> list:
        """
        Locate figures in the PDF as
        {figure_index, pdf_page (1-indexed), bbox, caption}. Deterministic —
        bbox from the image placement, caption from the nearest
        "Figure/Table ..." text block. Sub-50pt images (icons/rules) are
        skipped. Returns [] on failure or when the PDF embeds no raster
        images (vector-only diagrams are a later enhancement). Page numbers
        share the 1-indexed space used everywhere else.
        """
        import pymupdf

        figures = []
        try:
            doc = pymupdf.open(filepath)
            fig_index = 0
            for pno in range(len(doc)):
                page = doc[pno]
                try:
                    infos = page.get_image_info(xrefs=True)
                except Exception:
                    infos = page.get_image_info()
                blocks = page.get_text("blocks")
                for info in infos:
                    bbox = info.get("bbox")
                    if not bbox:
                        continue
                    x0, y0, x1, y1 = bbox
                    if (
                        (x1 - x0) < self._MIN_FIGURE_PT
                        or (y1 - y0) < self._MIN_FIGURE_PT
                    ):
                        continue
                    figures.append(
                        {
                            "figure_index": fig_index,
                            "pdf_page": pno + 1,
                            "bbox": [round(float(c), 2) for c in bbox],
                            "caption": self._nearest_caption(bbox, blocks),
                        }
                    )
                    fig_index += 1
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to extract figures: {e}")
            return []

        logger.info(f"Extracted figures: {len(figures)}")
        return figures

    def _nearest_caption(self, bbox, blocks):
        """Nearest text block that reads like a caption (starts with
        Figure/Table/...), horizontally overlapping the image and within
        120pt vertically. None if there's no plausible caption."""
        x0, y0, x1, y1 = bbox
        best = None
        best_dist = 1e9
        for blk in blocks:
            if len(blk) < 5:
                continue
            bx0, by0, bx1, by1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
            if not text or not self._CAPTION_RE.match(text):
                continue
            if bx1 < x0 or bx0 > x1:  # no horizontal overlap
                continue
            if by0 >= y1:  # below the image
                dist = by0 - y1
            elif by1 <= y0:  # above the image
                dist = y0 - by1
            else:  # vertically overlapping
                dist = 0.0
            if dist < best_dist and dist < 120.0:
                best_dist = dist
                best = " ".join(text.split())
        return best


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
        toc: Optional[list] = None,
        exam_mode: bool = False,
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
        # Exam papers segment by QUESTION, not by markdown header — a whole
        # question (stem + its sub-parts) is the coherent retrievable unit;
        # 512-token windows would shred the shared scenario/data a question's
        # parts depend on. Falls through to the generic path if segmentation
        # can't find a reliable question structure (returns None).
        exam_chunks = self._split_exam_questions(text) if exam_mode else None

        if exam_chunks is not None:
            # Intentionally NOT token-capped: keep each question whole even if
            # it exceeds max_chunk_tokens. (Trade-off: a very long question may
            # exceed the embed model's context and get truncated at embed time
            # — acceptable for exam questions, which are typically short.)
            header_processed_chunks = exam_chunks
        else:
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
            # Exam segmentation stamps chunk_type ("question"/"preamble") and
            # question fields on the metadata; the generic path leaves them
            # unset and keeps the original header/recursive classification.
            chunk_type = chunk.metadata.get("chunk_type") or (
                "recursive" if parent_idx is not None else "header"
            )
            question_number = chunk.metadata.get("question_number")
            marks = chunk.metadata.get("marks")
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
            # Only present for exam chunks — keeps generic annotation blocks
            # byte-identical to before (byte-offset math is unaffected either
            # way since metadata_size is measured from the real block).
            if question_number is not None:
                header_meta_lines.append(f"question_number: {question_number}")
            if marks is not None:
                header_meta_lines.append(f"marks: {marks}")
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

            # Structural breadcrumb from the TOC, keyed on the chunk's start
            # page. Deterministic — no LLM — and null when the PDF had no
            # embedded outline.
            section_path_list = resolve_toc_path(toc, page_start) if toc else []
            section_path_json = json.dumps(section_path_list)
            chapter_title = (
                section_path_list[0]["title"] if section_path_list else None
            )
            section_title = (
                section_path_list[-1]["title"] if section_path_list else None
            )
            # Structural zone (front_matter / body / glossary / index /
            # appendix / ...), derived deterministically from the breadcrumb.
            zone = classify_zone(section_path_list)

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
            chunk.metadata["section_path"] = section_path_list
            chunk.metadata["section_title"] = section_title
            chunk.metadata["chapter_title"] = chapter_title
            chunk.metadata["question_number"] = question_number
            chunk.metadata["marks"] = marks
            chunk.metadata["zone"] = zone

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
                    section_path_json,
                    section_title,
                    chapter_title,
                    question_number,
                    marks,
                    zone,
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

    # ------------------------------------------------------------------
    # Exam-paper segmentation (doc_type == "exam_paper")
    # ------------------------------------------------------------------
    # Leading char class absorbs markdown noise pymupdf4llm emits around a
    # heading (bold "**", blockquote ">", heading "#", underline "_").
    _QUESTION_HEADER_RE = re.compile(
        r"^[ \t>*_#]*(?:QUESTION|Question)[ \t]+(\d+)\b.*$", re.MULTILINE
    )
    # Fallback for papers that number top-level questions as "1." / "2)"
    # rather than "QUESTION 1". Only trusted when the numbers run 1,2,3…
    _NUMERIC_HEADER_RE = re.compile(r"^[ \t>*_#]*(\d+)[.)][ \t]+\S", re.MULTILINE)
    # End-of-line sub-part marks: "(3)" possibly wrapped in bold markers.
    _MARK_PAREN_RE = re.compile(
        r"[（(]\s*(\d{1,3})\s*[)）][ \t*_]*$", re.MULTILINE
    )
    # Explicit bracketed question total: "[25]".
    _MARK_TOTAL_RE = re.compile(r"[\[【]\s*(\d{1,3})\s*[\]】]")

    def _split_exam_questions(self, text):
        """
        Segment an exam paper into one Document per top-level question (stem
        + all its sub-parts kept together), in document order. A leading
        "preamble" Document captures any cover-page/instructions before Q1 so
        the segmentation covers the full text and the assembly loop's byte/
        page math stays aligned.

        Returns None when no reliable question structure is found — the
        caller then falls back to the generic header/recursive chunker.
        """
        matches = list(self._QUESTION_HEADER_RE.finditer(text))
        if len(matches) < 2:
            candidate = list(self._NUMERIC_HEADER_RE.finditer(text))
            numbers = [int(m.group(1)) for m in candidate]
            # Guard against incidental "1." list items: only accept the
            # numeric fallback when it's a clean 1..N run.
            if len(candidate) >= 2 and numbers == list(range(1, len(numbers) + 1)):
                matches = candidate
            else:
                return None

        docs: list[Document] = []

        preamble = text[: matches[0].start()]
        if preamble.strip():
            docs.append(
                Document(
                    page_content=preamble.strip("\n"),
                    metadata={
                        "chunk_type": "preamble",
                        "question_number": None,
                        "marks": None,
                    },
                )
            )

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip("\n")
            docs.append(
                Document(
                    page_content=block,
                    metadata={
                        "chunk_type": "question",
                        "question_number": m.group(1),
                        "marks": self._sum_marks(block),
                    },
                )
            )

        return docs

    def _sum_marks(self, block: str):
        """
        Best-effort total marks for a question block. Prefers an explicit
        bracketed total (e.g. "[25]"); otherwise sums end-of-line sub-part
        marks (e.g. "(3)" after each answer line). None when neither is
        present. Heuristic — mark conventions vary by paper.
        """
        totals = [int(x) for x in self._MARK_TOTAL_RE.findall(block)]
        if totals:
            return max(totals)
        parens = [int(x) for x in self._MARK_PAREN_RE.findall(block)]
        if parens:
            return sum(parens)
        return None

    # ------------------------------------------------------------------
    # Glossary parsing (zone == "glossary")
    # ------------------------------------------------------------------
    # "Term: definition" / "Term — definition" / "**Term** definition".
    # Term is short-ish and starts with a letter; the separator is a colon or
    # a dash. Bold/list markers around the term are tolerated and stripped.
    _GLOSSARY_ENTRY_RE = re.compile(
        r"^[ \t>*_-]*"
        r"(?:\*{1,2}|_)?(?P<term>[A-Za-z][A-Za-z0-9 ,'/()\-]{1,58}?)(?:\*{1,2}|_)?"
        r"[ \t]*[:—–-][ \t]+"
        r"(?P<def>\S.*)$"
    )

    def _parse_glossary_entries(self, text: str):
        """
        Parse "term : definition" / "term — definition" lines from a glossary
        block into (term, definition) pairs. Definitions that wrap onto
        following (non-entry, non-blank) lines are joined onto the current
        entry. Heuristic — glossary layouts vary; unmatched lines are ignored.
        """
        entries: list[tuple[str, str]] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            m = self._GLOSSARY_ENTRY_RE.match(line)
            if m:
                term = m.group("term").strip()
                definition = m.group("def").strip()
                entries.append((term, definition))
            elif entries and line.strip():
                # Continuation of the previous definition.
                term, definition = entries[-1]
                entries[-1] = (term, f"{definition} {line.strip()}")
        return entries


# Structural-zone rules, checked in priority order against the TOC breadcrumb
# titles. Back-matter keywords anchor at the title START (these sections lead
# with the keyword: "Glossary", "Appendix A", "References"). The ambiguous
# "index" is matched only as a WHOLE title ("Index" / "Subject Index") so a
# chapter titled "Index of Refraction" or "Index Numbers" stays body.
_ZONE_RULES = [
    ("glossary", re.compile(r"^\s*glossary\b", re.I)),
    ("index", re.compile(r"^\s*(?:subject|author|name)?\s*index\s*$", re.I)),
    (
        "bibliography",
        re.compile(
            r"^\s*(bibliography|references|works cited|further reading)\b", re.I
        ),
    ),
    ("appendix", re.compile(r"^\s*(appendix|appendices)\b", re.I)),
    (
        "answers",
        re.compile(
            r"^\s*(answers?|solutions?|answer key|memorandum|marking guideline)\b",
            re.I,
        ),
    ),
    (
        "front_matter",
        re.compile(
            r"^\s*(contents|table of contents|preface|foreword|"
            r"acknowledge?ments?|copyright|dedication|about the author|"
            r"frontispiece)\b",
            re.I,
        ),
    ),
]


def classify_zone(section_path):
    """
    Map a TOC breadcrumb (list of {"level","title"}) to a structural zone:
    front_matter / body / glossary / index / bibliography / appendix /
    answers. Deterministic — no LLM. Defaults to "body" when nothing in the
    breadcrumb matches a known zone keyword (or when there's no breadcrumb,
    e.g. a PDF with no embedded outline).
    """
    if not section_path:
        return "body"
    titles = [n.get("title", "") for n in section_path]
    for zone, pattern in _ZONE_RULES:
        if any(pattern.search(t) for t in titles):
            return zone
    return "body"


def resolve_toc_path(toc, page):
    """
    Given a flat TOC (list of [level, title, page], 1-indexed, document
    order) and a page number, return the breadcrumb of headings that
    enclose that page — outermost (chapter) to innermost (deepest
    subsection) — as a list of {"level", "title"} dicts.

    Resolution rule: the section a page belongs to is the deepest heading
    whose start page is at or before it; its ancestors are recovered by
    keeping a level stack (pop entries whose level is >= the current one,
    then push). Returns [] when there is no TOC or nothing precedes the
    page (e.g. front matter before the first heading).
    """
    if not toc or page is None:
        return []

    stack: list[dict] = []
    for entry in toc:
        level, title, entry_page = entry[0], entry[1], entry[2]
        if entry_page is None or entry_page < 1:
            continue
        if entry_page > page:
            # TOC is in document order — once we pass the page, we're done.
            break
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        stack.append({"level": level, "title": title})

    return stack
