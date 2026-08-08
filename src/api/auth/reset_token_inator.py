"""
api.auth.reset_token_inator
===========================
Primitives for the forgotten-password flow:

  * a 6-digit one-time shortcode emailed to the user (hashed, keyed with the
    signing secret, before storage — never stored in the clear);
  * a signed, short-lived RESET TOKEN handed back after the code verifies, which
    the update-password step presents (so the code is proven once, not twice).

Both hang off the same signing secret as the bearer tokens (token_inator), so
there's no new dependency or config.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from api.auth.token_inator import get_token_secret

_SALT = "cerebrum-pw-reset-v1"

CODE_TTL_SECONDS = 15 * 60          # shortcode validity window
RESET_TOKEN_TTL_SECONDS = 15 * 60   # verify → update window
MAX_CODE_ATTEMPTS = 5               # wrong guesses before a code locks


def generate_shortcode() -> str:
    """A 6-digit numeric one-time code (leading zeros kept)."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_code(code: str) -> str:
    """Keyed SHA-256 of the code. Keying with the signing secret means a DB leak
    alone can't brute-force the tiny 6-digit space."""
    return hashlib.sha256(f"{get_token_secret()}:{code}".encode("utf-8")).hexdigest()


def verify_code(code: str, code_hash: str) -> bool:
    return hmac.compare_digest(hash_code(code), code_hash)


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_token_secret(), salt=_SALT)


def mint_reset_token(user_id: str, reset_id: str) -> str:
    """Issue the interim reset token after a code verifies."""
    return _serializer().dumps({"uid": user_id, "rid": reset_id})


def read_reset_token(token: str) -> Optional[dict]:
    """Return {'uid', 'rid'} from a valid, unexpired reset token, else None."""
    if not token:
        return None
    try:
        data = _serializer().loads(token, max_age=RESET_TOKEN_TTL_SECONDS)
        return data if isinstance(data, dict) else None
    except (BadSignature, SignatureExpired):
        return None
