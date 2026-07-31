# cerebrum_core/utils/user_context_inator.py
from fastapi import Header, HTTPException, Request


def get_current_user_id(
    request: Request,
    x_user_id: str | None = Header(default=None),
) -> str:
    """
    Single source of truth for 'who is making this request', used by
    every router (learn, study_plan, bubble if it ever needs it).
    Validates against the users table so a stale/deleted user_id from
    a reinstalled client fails loudly instead of silently writing
    orphaned rows.
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id header")

    repo = request.app.state.note_registry
    if not repo.get_user(x_user_id):
        raise HTTPException(status_code=404, detail=f"Unknown user: {x_user_id}")

    return x_user_id
