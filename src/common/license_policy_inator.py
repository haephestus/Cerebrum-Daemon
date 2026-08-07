"""
common.license_policy_inator
============================
Decide whether a suggested reading's content may be **ingested** (fetched,
embedded, stored in the KB) or only surfaced as a **pointer** (title/link),
given its license and the deployment's commercial policy.

Load-bearing rule for gap 3: "free to read" != "free to use commercially — the
license decides." Ingestion is allowed only when the license clears policy:

  commercial build (default):  CC-BY / CC-BY-SA / CC0 / public-domain  -> ingest
                               CC-BY-NC / CC-BY-ND / unknown / free-read -> pointer
  non-commercial build:        additionally allows NC (NonCommercial)
                               ND / unknown / free-read                 -> pointer

`commercial_allowed` defaults from env CEREBRUM_COMMERCIAL_USE (default true,
since commercial use is in consideration).
"""

from __future__ import annotations

import os

# Normalized license tokens.
CC_BY = "cc-by"
CC_BY_SA = "cc-by-sa"
CC0 = "cc0"
PUBLIC_DOMAIN = "public-domain"
CC_BY_NC = "cc-by-nc"
CC_BY_ND = "cc-by-nd"
CC_BY_NC_SA = "cc-by-nc-sa"
CC_BY_NC_ND = "cc-by-nc-nd"
UNKNOWN = "unknown"

# Ingestable regardless of commercial policy (attribution/share-alike are
# redistribution obligations, handled by carrying the license in metadata).
_ALWAYS_INGEST = {CC_BY, CC_BY_SA, CC0, PUBLIC_DOMAIN}
# Ingestable ONLY on a non-commercial build.
_NC_INGEST = {CC_BY_NC, CC_BY_NC_SA}
# ND forbids derivatives (chunking/embedding is a derivative) — never ingest.
_NEVER_INGEST = {CC_BY_ND, CC_BY_NC_ND}


def commercial_allowed() -> bool:
    return os.getenv("CEREBRUM_COMMERCIAL_USE", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def normalize(license_str: str | None) -> str:
    """Map a free-form license string to a normalized token. Unrecognised or
    empty -> UNKNOWN (treated as pointer-only — fail safe)."""
    if not license_str:
        return UNKNOWN
    s = license_str.strip().lower()
    s = s.replace("licence", "license")
    # public domain / CC0
    if "cc0" in s or "public domain" in s or s in ("pd", "publicdomain"):
        return CC0 if "cc0" in s else PUBLIC_DOMAIN
    if "creativecommons" in s or "creative commons" in s or s.startswith("cc"):
        nc = "nc" in s or "noncommercial" in s or "non-commercial" in s
        nd = "nd" in s or "noderiv" in s
        sa = "sa" in s or "sharealike" in s or "share-alike" in s
        if nc and nd:
            return CC_BY_NC_ND
        if nc and sa:
            return CC_BY_NC_SA
        if nc:
            return CC_BY_NC
        if nd:
            return CC_BY_ND
        if sa:
            return CC_BY_SA
        return CC_BY
    return UNKNOWN


def is_ingestable(license_str: str | None, commercial: bool | None = None) -> bool:
    if commercial is None:
        commercial = commercial_allowed()
    tok = normalize(license_str)
    if tok in _NEVER_INGEST:
        return False
    if tok in _ALWAYS_INGEST:
        return True
    if tok in _NC_INGEST:
        return not commercial  # NC ok only when NOT commercial
    return False  # UNKNOWN / free-read -> pointer


def decide(license_str: str | None, commercial: bool | None = None) -> tuple[str, str]:
    """('ingest'|'pointer', reason) for a candidate's license."""
    if commercial is None:
        commercial = commercial_allowed()
    tok = normalize(license_str)
    if is_ingestable(license_str, commercial):
        return "ingest", f"license {tok} permits {'commercial ' if commercial else ''}reuse"
    if tok == UNKNOWN:
        return "pointer", "license unknown/not open — surfaced as a pointer only"
    if tok in _NEVER_INGEST:
        return "pointer", f"license {tok} forbids derivatives (no chunk/embed)"
    return "pointer", f"license {tok} not permitted under current commercial policy"
