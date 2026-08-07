"""
api.auth.token_inator
======================
Per-user bearer tokens — the "user API" that replaces trust-the-header identity
in cloud mode (and is available in local mode too). A token is an itsdangerous
signed, timed blob carrying the user_id; verifying it needs no DB round-trip and
no extra dependency (itsdangerous ships already).

Signing secret:
  * cloud: CEREBRUM_TOKEN_SECRET env var — MUST be set and identical across every
    instance, or one instance can't verify a token another minted. We fail loudly
    rather than silently minting unverifiable tokens.
  * local: falls back to the daemon key file, which is stable per machine.

Rotating the secret invalidates every outstanding token — that's the revoke-all
lever. For per-token revocation, add a denylist table later; deliberately not
built yet (YAGNI for the current scale).
"""

import os
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from common.deploy_config_inator import TOKEN_MAX_AGE_SECONDS, is_cloud

_SALT = "cerebrum-user-token-v1"


def get_token_secret() -> str:
    env = os.getenv("CEREBRUM_TOKEN_SECRET")
    if env:
        return env
    if is_cloud():
        raise RuntimeError(
            "CEREBRUM_TOKEN_SECRET must be set in cloud mode (it has to be shared "
            "across instances so any of them can verify a token)."
        )
    # Local single-machine fallback: reuse the daemon key as the signing secret.
    from api.auth.daemon_auth_inator import get_or_create_daemon_key

    return get_or_create_daemon_key()


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_token_secret(), salt=_SALT)


def mint_token(user_id: str) -> str:
    """Issue a signed token for a just-authenticated user."""
    return _serializer().dumps(user_id)


def read_token(token: str) -> Optional[str]:
    """Return the user_id from a valid, unexpired token, else None. Never raises
    on a bad/expired/tampered token — callers treat None as 'not authenticated'."""
    if not token:
        return None
    try:
        return _serializer().loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
