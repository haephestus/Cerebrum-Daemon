# api/auth/daemon_auth_inator.py
"""
api.auth.daemon_auth_inator
================================
Minimal request-identity extraction. There's no password/session layer
yet -- the Flutter client sends its locally-saved user_id as a header,
and we trust it for now. This is a placeholder for real auth (a signed
token would replace the header value, not the mechanism) but it's
enough to stop routes from silently operating on/leaking ANY user's
data regardless of who's asking, which is the actual bug being fixed.
"""
import secrets

from common.file_util_inator import CerebrumPaths

_KEY_FILE = CerebrumPaths().config_root_dir() / "daemon_api_key.txt"


def get_or_create_daemon_key() -> str:
    """
    One static key for the whole daemon, generated on first run and
    reused forever after. This is NOT per-user auth — it's the gate
    that keeps the ngrok tunnel from being wide open to the internet.
    """
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()

    key = secrets.token_urlsafe(32)
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_text(key)
    _KEY_FILE.chmod(0o600)  # owner-read-only, since this is a bearer secret
    return key
