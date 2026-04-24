from fastapi import APIRouter, HTTPException

from cerebrum_core.model_inator import UserConfig
from cerebrum_core.user_inator import ConfigManager

configs_router = APIRouter(prefix="/user", tags=["user-config"])

config = ConfigManager()


# ─────────────────────────────────────────────────────────────
# GET full user config
# ─────────────────────────────────────────────────────────────
@configs_router.get("/config", response_model=UserConfig)
def get_user_config():
    try:
        return config.load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET installed chat models
# ─────────────────────────────────────────────────────────────
@configs_router.get("/models/chat/installed")
def list_installed_chat_models():
    try:
        chat, _ = config.get_installed_models()
        return {"installed_chat_models": chat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET installed embedding models
# ─────────────────────────────────────────────────────────────
@configs_router.get("/models/embedding/installed")
def list_installed_embedding_models():
    try:
        _, emb = config.get_installed_models()
        return {"installed_embedding_models": emb}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET online models (full Ollama library)
# ─────────────────────────────────────────────────────────────
@configs_router.get("/models/online")
def list_online_models():
    try:
        return config.get_available_online_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST update chat model only
# ─────────────────────────────────────────────────────────────
@configs_router.post("/config/models/chat", response_model=UserConfig)
def update_chat_model(chat_model: str):
    try:
        return config.update_model_settings(chat=chat_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST update embedding model only
# ─────────────────────────────────────────────────────────────
@configs_router.post("/config/models/embedding", response_model=UserConfig)
def update_embedding_model(embedding_model: str):
    try:
        return config.update_model_settings(embedding=embedding_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST download model
# ─────────────────────────────────────────────────────────────
@configs_router.post("/models/download/{model_name}")
def download_model(model_name: str):
    try:
        config.download_model(model_name)
        return {"message": f"Model '{model_name}' downloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET Ollama status
# ─────────────────────────────────────────────────────────────
@configs_router.get("/ollama/status")
def ollama_status():
    try:
        return config.get_ollama_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET model details (description and tags)
# ─────────────────────────────────────────────────────────────
@configs_router.get("/models/{model_name}/details")
def get_model_details(model_name: str):
    try:
        details = config.get_model_details(model_name)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
