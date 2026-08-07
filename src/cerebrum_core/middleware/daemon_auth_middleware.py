# cerebrum_core/middleware/daemon_auth_middleware.py
"""
Mode-aware request auth.

Whatever the mode, if a valid `Authorization: Bearer <token>` is present we
resolve it to a user_id and stash it on request.state.user_id, so downstream
get_current_user_id can trust it without re-reading the header.

  * local: transport is gated by the shared X-Daemon-Key (the tunnel guard).
           Identity still flows via X-User-Id (or a bearer token if the client
           chooses to log in). This is unchanged single-machine behaviour.

  * cloud: no shared key — every request MUST carry a valid bearer token, and
           the derived user_id is the identity. Missing/invalid token -> 401,
           before any route runs.
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.auth.daemon_auth_inator import get_or_create_daemon_key
from api.auth.token_inator import read_token
from cerebrum_core.deploy_config_inator import is_cloud

# Routes that don't need auth — health checks, docs, the liveness root, and the
# unauthenticated entry points (create an account / log in to GET a token).
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/user/account",
    "/user/login",
}


def _bearer_user_id(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return read_token(auth[7:].strip())
    return None


class DaemonAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Let CORS preflight through untouched — the browser never attaches
        # custom headers (incl. X-Daemon-Key / Authorization) to an OPTIONS
        # preflight, so gating it here would 401 every cross-origin request.
        # /user/account is only public for POST (create); other methods on it
        # still require auth.
        public = request.url.path in PUBLIC_PATHS and not (
            request.url.path == "/user/account" and request.method != "POST"
        )
        if request.method == "OPTIONS" or public:
            return await call_next(request)

        # Resolve a bearer token if one is present, in either mode.
        user_id = _bearer_user_id(request)
        if user_id:
            request.state.user_id = user_id

        if is_cloud():
            if not user_id:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid bearer token"},
                )
            return await call_next(request)

        # local mode: the daemon key gates the transport.
        if request.headers.get("X-Daemon-Key") != get_or_create_daemon_key():
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing daemon key"}
            )
        return await call_next(request)
