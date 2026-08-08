"""
common.content_fetch_inator — fetch a reading's URL and extract clean markdown
for ingestion. PDF via pymupdf (already used for KB PDFs); HTML via trafilatura
(optional dependency — returns None with a log line if unavailable). Any failure
is swallowed to None so the accept flow can degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_UA = "Cerebrum-Daemon/dev (research/study assistant)"


def fetch_as_markdown(url: str, timeout: float = 25.0) -> Optional[str]:
    if not url:
        return None
    try:
        import httpx

        r = httpx.get(
            url, timeout=timeout, follow_redirects=True, headers={"User-Agent": _UA}
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("content fetch failed %s: %s", url, e)
        return None

    ctype = (r.headers.get("content-type") or "").lower()
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        return _pdf_to_markdown(r.content)
    return _html_to_markdown(r.text)


def _pdf_to_markdown(data: bytes) -> Optional[str]:
    try:
        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(stream=data, filetype="pdf")
        return pymupdf4llm.to_markdown(doc) or None
    except Exception as e:
        logger.warning("PDF→markdown failed: %s", e)
        return None


def _html_to_markdown(html: str) -> Optional[str]:
    try:
        import trafilatura
    except ImportError:
        logger.warning(
            "trafilatura not installed — cannot extract HTML readings "
            "(add it to requirements to enable HTML ingestion)"
        )
        return None
    try:
        return (
            trafilatura.extract(html, output_format="markdown", include_links=False)
            or None
        )
    except Exception as e:
        logger.warning("HTML→markdown failed: %s", e)
        return None
