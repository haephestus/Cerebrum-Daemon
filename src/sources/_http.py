"""
sources._http — tiny shared HTTP helper for the external providers.

Keyless GETs with a sane UA, redirect-follow, timeout, and swallow-to-None on
failure (a dead provider must never break a suggestion run — the KB tier and
other providers still return).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_UA = "Cerebrum-Daemon/dev (research/study assistant)"


def get_json(
    url: str, params: Optional[dict] = None, timeout: float = 8.0
) -> Optional[Any]:
    try:
        import httpx

        r = httpx.get(
            url,
            params=params,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA, "Accept": "application/json"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:  # network down, rate limit, bad json, offline...
        logger.warning("provider GET %s failed: %s", url, e)
        return None
