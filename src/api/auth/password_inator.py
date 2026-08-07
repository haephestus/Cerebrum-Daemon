"""
api.auth.password_inator
========================
bcrypt password hashing/verification. Kept separate from the repository so
the DB layer only ever stores/reads an opaque `password_hash` string and never
touches the plaintext — hashing lives at the API edge (account create/login).
"""
import bcrypt

# bcrypt hard-caps the input at 72 bytes and silently ignores the rest, which
# would make two different long passwords collide. We reject rather than
# truncate so the limit is explicit to the caller.
_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """Return a bcrypt hash (salt embedded) for storage in users.password_hash."""
    pw = password.encode("utf-8")
    if len(pw) > _MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password too long ({len(pw)} bytes); max is {_MAX_PASSWORD_BYTES}."
        )
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """True iff `password` matches the stored hash. A blank hash (legacy rows
    from migration 0003, or an unset account) never matches."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), password_hash.encode("utf-8")
        )
    except ValueError:
        # Malformed/non-bcrypt stored value — treat as no match, not a crash.
        return False
