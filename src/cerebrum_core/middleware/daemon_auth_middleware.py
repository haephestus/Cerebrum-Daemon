# cerebrum_core/middleware/daemon_auth_middleware.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.auth.daemon_auth_inator import get_or_create_daemon_key

# Routes that don't need the key — health checks, etc.
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class DaemonAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        expected = get_or_create_daemon_key()
        provided = request.headers.get("X-Daemon-Key")

        if provided != expected:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing daemon key"}
            )

        return await call_next(request)
