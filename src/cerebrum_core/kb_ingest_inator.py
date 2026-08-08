"""
cerebrum_core.kb_ingest_inator — graduate an external reading into the KB.

Reuses the exact chunk + embed components the PDF-upload path uses, but feeds
markdown directly (skipping the PDF→markdown converter). Provenance (source /
url / license) is written into the markdown header so it travels with the
content. register_inator dedups by fingerprint, so re-ingesting the same reading
is a no-op. The embed step needs a live embedding model (Ollama), same as PDF
upload — so this is verifiable only against a running stack.
"""

from __future__ import annotations

import logging
import re

from common.file_util_inator import CerebrumPaths

logger = logging.getLogger(__name__)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "doc").lower()).strip("-")[:60] or "doc"


def ingest_external_document(
    *,
    markdown_text: str,
    title: str,
    file_registry,
    source: str,
    url: str | None = None,
    license: str | None = None,
    domain: str = "external",
    doc_type: str = "book",
) -> str:
    """Write external markdown → register → chunk → embed into the KB. Returns
    the file_fingerprint. `file_registry` is a FileRegisterInator."""
    from cerebrum_core.knowledgebase_inator import FileMarkdownChunker
    from vectorstore.embedd_inator import EmbeddInator

    subject = _slug(title)
    art_dir = CerebrumPaths().kb_root_dir() / "external" / _slug(source)
    art_dir.mkdir(parents=True, exist_ok=True)
    md_path = art_dir / f"{subject}.md"
    header = f"<!-- source: {source} | url: {url or ''} | license: {license or 'unknown'} -->\n\n"
    md_path.write_text(header + (markdown_text or ""), encoding="utf-8")

    fingerprint = file_registry.register_inator(
        original_name=title, filepath=str(md_path)
    )
    file_registry.mark_converted_inator(
        file_fingerprint=fingerprint,
        domain=domain,
        subject=subject,
        sanitized_name=title,
        doc_type=doc_type,
    )
    chunked_path = FileMarkdownChunker().chunk(
        markdown_path=md_path, file_fingerprint=fingerprint, doc_type=doc_type
    )
    EmbeddInator(
        original_name=title, file_fingerprint=fingerprint
    ).embed_from_chunked_markdown(
        chunked_markdown=chunked_path,
        collection_name=subject,
        domain=domain,
        subject=subject,
    )
    file_registry.mark_embedded_inator(file_fingerprint=fingerprint)
    logger.info("ingested external reading %r as %s", title, fingerprint)
    return fingerprint
