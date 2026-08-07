"""
cerebrum_core.deploy_config_inator
===================================
Single switch for how the daemon is being run:

  * "local"  — your own machine / a tunnel. One shared X-Daemon-Key gates the
               transport; identity comes from X-User-Id (trusted, it's your box).
               Background workers run in-process; state lives in local SQLite/FS.

  * "cloud"  — hosted, multi-tenant (e.g. Leapcell). No shared daemon key; every
               request must carry a per-user bearer token. Background workers and
               local-disk state are NOT assumed here (see the storage plan).

Set CEREBRUM_DEPLOYMENT_MODE=cloud to flip it; defaults to local so nothing
changes for existing single-user setups. This module intentionally imports
nothing from `api` — the api layer depends on it, not the other way round.
"""

import os

_LOCAL = "local"
_CLOUD = "cloud"

# How long a login token stays valid (seconds). 30 days by default — a learning
# app doesn't need aggressive expiry, and tokens are revocable in cloud mode via
# rotating CEREBRUM_TOKEN_SECRET if needed.
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 30


def deployment_mode() -> str:
    mode = os.getenv("CEREBRUM_DEPLOYMENT_MODE", _LOCAL).strip().lower()
    return mode if mode in (_LOCAL, _CLOUD) else _LOCAL


def is_cloud() -> bool:
    return deployment_mode() == _CLOUD


def is_local() -> bool:
    return not is_cloud()
