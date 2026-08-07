# cerebrum_core/utils/user_context_inator.py
from fastapi import Header, HTTPException, Request


def get_current_user_id(
    request: Request,
    x_user_id: str | None = Header(default=None),
) -> str:
    """
    Single source of truth for 'who is making this request', used by
    every router (learn, study_plan, bubble if it ever needs it).

    Precedence: a bearer token verified by DaemonAuthMiddleware (which sets
    request.state.user_id) wins; otherwise, in local mode, the X-User-Id
    header is trusted. Cloud mode never reaches the header fallback because the
    middleware already rejected any request without a valid token.

    Validates against the users table so a stale/deleted user_id from a
    reinstalled client fails loudly instead of silently writing orphaned rows.
    """
    user_id = getattr(request.state, "user_id", None) or x_user_id
    if not user_id:
        raise HTTPException(
            status_code=401, detail="Missing identity (bearer token or X-User-Id)"
        )

    repo = request.app.state.note_registry
    if not repo.get_user(user_id):
        raise HTTPException(status_code=404, detail=f"Unknown user: {user_id}")

    return user_id
