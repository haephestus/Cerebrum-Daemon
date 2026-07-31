import sqlite3
import uuid
from typing import Optional

from fastapi import Request  # add to the existing fastapi import line
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from cerebrum_core.model_inator import UserConfig
from cerebrum_core.user_inator import ConfigManager

# Import the generator to handle CLI statuses and downloads transparently
from cerebrum_core.utils.ollama_compat.ollama_parser_inator import (
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
    email: EmailStr
    settings: Optional[dict] = None


@user_router.post("/account")
def create_account(body: AccountCreateRequest, request: Request):
    repo = request.app.state.note_registry
    user_id = uuid.uuid4().hex
    try:
        repo.create_user(
            user_id=user_id,
            name=body.name,
            email=body.email,
            settings=body.settings,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409, detail=f"Email already registered: {body.email}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"id": user_id, "name": body.name, "email": body.email}


# ─────────────────────────────────────────────────────────────
# GET account by id
# ─────────────────────────────────────────────────────────────
@user_router.get("/account/{user_id}")
def get_account(user_id: str, request: Request):
    repo = request.app.state.note_registry
    try:
        user = repo.get_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


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
