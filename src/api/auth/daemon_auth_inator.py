# cerebrum_core/utils/daemon_auth_inator.py
import secrets

from cerebrum_core.utils.file_util_inator import CerebrumPaths

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
