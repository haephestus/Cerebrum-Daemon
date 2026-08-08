import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from api.auth.email_inator import get_email_sender
from api.auth.password_inator import hash_password, verify_password
from api.auth.reset_token_inator import (
    CODE_TTL_SECONDS,
    MAX_CODE_ATTEMPTS,
    generate_shortcode,
    hash_code,
    mint_reset_token,
    read_reset_token,
    verify_code,
)
from api.auth.token_inator import mint_token
from models.model_inator import UserConfig
from cerebrum_core import learning_profile_inator as lp
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.user_context_inator import build_effective_profile, get_current_user_id

# Import the generator to handle CLI statuses and downloads transparently
from common.ollama_compat.ollama_parser_inator import (
    OllamaManifestGenerator,
)

user_router = APIRouter(prefix="/user", tags=["user-config"])

config = ConfigManager()
ollama_engine = OllamaManifestGenerator()


# ─────────────────────────────────────────────────────────────
# POST create account (engrams/learning-center identity)
# ─────────────────────────────────────────────────────────────
class AccountCreateRequest(BaseModel):
    name: str
    email: EmailStr  # mandatory: it's the login identifier and is UNIQUE in the DB
    password: str = Field(min_length=8)
    settings: Optional[dict] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@user_router.post("/account")
def create_account(body: AccountCreateRequest, request: Request):
    repo = request.app.state.note_registry
    user_id = uuid.uuid4().hex
    try:
        repo.create_user(
            user_id=user_id,
            name=body.name,
            email=body.email,
            password_hash=hash_password(body.password),
            settings=body.settings,
        )
    except sqlite3.IntegrityError:
        # email is UNIQUE NOT NULL in the users table, so this fires when an
        # account with the same email already exists.
        raise HTTPException(
            status_code=409, detail=f"Account already exists: {body.email}"
        )
    except ValueError as e:
        # e.g. password over bcrypt's 72-byte limit
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"id": user_id, "name": body.name, "email": body.email}


# ─────────────────────────────────────────────────────────────
# POST login -- verify email + password, hand back the user_id the
# client then sends as X-User-Id on every subsequent request
# ─────────────────────────────────────────────────────────────
@user_router.post("/login")
def login(body: LoginRequest, request: Request):
    repo = request.app.state.note_registry
    user = repo.get_user_by_email(body.email)
    # Verify even when the user is missing (against a blank hash) so the
    # response time doesn't reveal whether the email exists.
    if not verify_password(body.password, (user or {}).get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Bearer token the client sends as `Authorization: Bearer <token>` on every
    # subsequent request — this is the identity in cloud mode, and works in
    # local mode too (alongside X-User-Id).
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "token": mint_token(user["id"]),
    }


# ─────────────────────────────────────────────────────────────
# POST change password -- authenticated: caller proves the current
# password, then sets a new one. (A forgotten-password email reset is a
# separate flow that needs email delivery, which the daemon doesn't have.)
# ─────────────────────────────────────────────────────────────
class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


@user_router.post("/password")
def change_password(
    body: PasswordChangeRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    user = repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    if not verify_password(body.current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    if body.new_password == body.current_password:
        raise HTTPException(
            status_code=400, detail="New password must differ from the current one"
        )

    try:
        new_hash = hash_password(body.new_password)
    except ValueError as e:
        # e.g. password over bcrypt's 72-byte limit
        raise HTTPException(status_code=400, detail=str(e))

    repo.update_password(user_id, new_hash)
    # Hand back a fresh token for convenience (existing tokens aren't tied to
    # the password, so they remain valid — token revocation is out of scope).
    return {"status": "password updated", "token": mint_token(user_id)}


# ─────────────────────────────────────────────────────────────
# Forgotten-password reset (unauthenticated — see PUBLIC_PATHS).
# request → emailed shortcode → verify (→ reset token) → update.
# ─────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    email: EmailStr


class ResetVerify(BaseModel):
    email: EmailStr
    code: str


class ResetUpdate(BaseModel):
    reset_token: str
    new_password: str = Field(min_length=8)


def _reset_expired(expires_at: str) -> bool:
    try:
        return datetime.utcnow() > datetime.fromisoformat(expires_at)
    except Exception:
        return True  # unparseable → treat as expired (fail safe)


@user_router.post("/password/reset-request")
def password_reset_request(body: ResetRequest, request: Request):
    """Start a reset. ALWAYS 200 — never reveals whether the email has an
    account. If it does, emails a one-time shortcode."""
    repo = request.app.state.note_registry
    user = repo.get_user_by_email(body.email)
    if user:
        code = generate_shortcode()
        expires = (
            datetime.utcnow() + timedelta(seconds=CODE_TTL_SECONDS)
        ).isoformat()
        repo.create_reset_code(user["id"], hash_code(code), expires)
        link_base = os.getenv("RESET_LINK_BASE", "cerebrum://reset-password")
        message = (
            f"Your Cerebrum password reset code is: {code}\n\n"
            f"It expires in 15 minutes. Open the app to continue:\n"
            f"{link_base}?email={quote(body.email)}\n\n"
            f"If you didn't request this, you can safely ignore this email."
        )
        try:
            get_email_sender().send(
                body.email, "Your Cerebrum password reset code", message
            )
        except Exception:
            pass  # never surface delivery state to the caller (anti-enumeration)
    return {"status": "If that email has an account, a reset code has been sent."}


@user_router.post("/password/reset-verify")
def password_reset_verify(body: ResetVerify, request: Request):
    """Verify the emailed code; on success return a short-lived reset token the
    update step presents. Generic errors so nothing leaks."""
    repo = request.app.state.note_registry
    generic = HTTPException(status_code=400, detail="Invalid or expired code")
    user = repo.get_user_by_email(body.email)
    if not user:
        raise generic
    reset = repo.get_active_reset(user["id"])
    if not reset:
        raise generic
    if _reset_expired(reset["expires_at"]) or reset["attempts"] >= MAX_CODE_ATTEMPTS:
        repo.mark_reset_used(reset["id"])
        raise generic
    if not verify_code(body.code, reset["code_hash"]):
        if repo.increment_reset_attempts(reset["id"]) >= MAX_CODE_ATTEMPTS:
            repo.mark_reset_used(reset["id"])  # lock after too many guesses
        raise generic
    return {"reset_token": mint_reset_token(user["id"], reset["id"])}


@user_router.post("/password/reset-update")
def password_reset_update(body: ResetUpdate, request: Request):
    """Set the new password using the reset token from verify. Burns the code."""
    repo = request.app.state.note_registry
    invalid = HTTPException(status_code=400, detail="Invalid or expired reset token")
    data = read_reset_token(body.reset_token)
    if not data:
        raise invalid
    uid, rid = data.get("uid"), data.get("rid")
    reset = repo.get_reset(rid) if rid else None
    if (
        not reset
        or reset["user_id"] != uid
        or reset["used_at"] is not None
        or _reset_expired(reset["expires_at"])
    ):
        raise invalid
    try:
        new_hash = hash_password(body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    repo.update_password(uid, new_hash)
    repo.invalidate_user_resets(uid)  # single-use: burn this + any siblings
    return {"status": "password updated", "token": mint_token(uid)}


# ─────────────────────────────────────────────────────────────
# DELETE my account -- wipes the caller and everything they own
# (notes, engrams, attempts, mastery, misconceptions, queue entries).
# Identity comes from X-User-Id, NOT a path param: you can only ever
# delete yourself, so no caller can wipe another user's account by id.
# ─────────────────────────────────────────────────────────────
@user_router.delete("/account")
def delete_account(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    repo = request.app.state.note_registry
    try:
        deleted = repo.delete_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    return {"deleted": True, "id": user_id}


# ─────────────────────────────────────────────────────────────
# GET my account -- identity from X-User-Id, so no one can read
# another user's account by guessing/knowing their id.
# ─────────────────────────────────────────────────────────────
@user_router.get("/account")
def get_account(request: Request, user_id: str = Depends(get_current_user_id)):
    repo = request.app.state.note_registry
    try:
        user = repo.get_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


# ─────────────────────────────────────────────────────────────
# Learning profile — the user's teaching preferences.
# GET returns the *effective* profile (declared prior blended with inferred
# evidence, confidence-gated) plus a prompt persona; PUT sets only the
# *declared* layer (inference is never client-writable).
# ─────────────────────────────────────────────────────────────
class LearningProfileUpdate(BaseModel):
    # axis-name -> scalar in [-1, 1]; unknown axes / out-of-range values are
    # dropped/clamped by learning_profile_inator.sanitize_declared.
    axes: dict[str, float]


@user_router.get("/learning-profile")
def get_learning_profile(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    repo = request.app.state.note_registry
    return build_effective_profile(repo, user_id)


@user_router.put("/learning-profile")
def put_learning_profile(
    body: LearningProfileUpdate,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    clean = lp.sanitize_declared(body.axes)
    repo.set_declared_profile(user_id, clean)
    return {"declared": clean, "effective": build_effective_profile(repo, user_id)}


# ─────────────────────────────────────────────────────────────
# GET full user config
# ─────────────────────────────────────────────────────────────
@user_router.get("/config", response_model=UserConfig)
def get_user_config():
    try:
        return config.load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET installed chat models
# ─────────────────────────────────────────────────────────────
@user_router.get("/models/chat/installed")
def list_installed_chat_models():
    try:
        # Route directly to our parser engine's local live API check
        chat, _ = ollama_engine.get_installed_models()
        return {"installed_chat_models": chat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET installed embedding models
# ─────────────────────────────────────────────────────────────
@user_router.get("/models/embedding/installed")
def list_installed_embedding_models():
    try:
        # Route directly to our parser engine's local live API check
        _, emb = ollama_engine.get_installed_models()
        return {"installed_embedding_models": emb}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET online models (Reads from local Master Manifest JSON)
# ─────────────────────────────────────────────────────────────
@user_router.get("/models/online")
def list_online_models():
    try:
        manifest = config.get_manifest_data()
        return {
            "online_chat_models": manifest.get("online_chat_models", []),
            "online_embedding_models": manifest.get("online_embedding_models", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST update chat model only
# ─────────────────────────────────────────────────────────────
@user_router.post("/config/models/chat", response_model=UserConfig)
def update_chat_model(chat_model: str):
    try:
        return config.update_model_settings(chat=chat_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST update cloud model only
# ─────────────────────────────────────────────────────────────
@user_router.post("/config/models/cloud", response_model=UserConfig)
def update_cloud_model(cloud_model: str):
    try:
        return config.update_model_settings(cloud_model=cloud_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST update embedding model only
# ─────────────────────────────────────────────────────────────
@user_router.post("/config/models/embedding", response_model=UserConfig)
def update_embedding_model(embedding_model: str):
    try:
        return config.update_model_settings(embedding=embedding_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST download model
# ─────────────────────────────────────────────────────────────
@user_router.post("/models/download/{model_name}")
def download_model(model_name: str):
    try:
        # Transferred work logic cleanly executed via CLI interaction instance
        ollama_engine.download_model(model_name)
        return {"message": f"Model '{model_name}' downloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET Ollama status
# ─────────────────────────────────────────────────────────────
@user_router.get("/ollama/status")
def ollama_status():
    try:
        # Transferred work logic cleanly mapped to system checking environment routines
        return ollama_engine.get_ollama_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET all cloud models
# ─────────────────────────────────────────────────────────────
@user_router.get("/models/cloud")
def list_cloud_models():
    try:
        manifest = config.get_manifest_data()
        cloud_models_map = manifest.get("cloud_models", {})
        return {"cloud_models": list(cloud_models_map.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET model details
# ─────────────────────────────────────────────────────────────
@user_router.get("/models/{model_name}/details")
def get_model_details(model_name: str):
    try:
        manifest = config.get_manifest_data()
        details = manifest.get("models_details", {}).get(model_name)
        if not details:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' data not synced in master manifest.",
            )
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET cloud tags for a specific model
# ─────────────────────────────────────────────────────────────
@user_router.get("/models/{model_name}/cloud-tags")
def get_cloud_tags(model_name: str):
    try:
        manifest = config.get_manifest_data()
        cloud_tags = manifest.get("cloud_models", {}).get(model_name, [])
        return {"cloud_tags": cloud_tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
