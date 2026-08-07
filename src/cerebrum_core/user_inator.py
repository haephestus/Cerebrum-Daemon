import json
from typing import Optional

from cerebrum_core.constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_EMBED_MODEL,
)
from cerebrum_core.model_inator import ModelConfig, UserConfig
from cerebrum_core.utils.file_util_inator import CerebrumPaths

CONFIG_ROOT = CerebrumPaths().config_root_dir()
CONFIG_FILE = CONFIG_ROOT / "user_config.json"
MANIFEST_FILE = CONFIG_ROOT / "models_manifest.json"


def should_use_cloud(config: Optional[UserConfig] = None) -> bool:
    """The one place the cloud-vs-local decision is made.

    The actual config field is `ollama.toggle_cloud`. Before this, callers
    read fields that don't exist on the model — study_planner checked
    `models.use_cloud` (always False) and ai_grading checked
    `ollama.prefer_cloud` (always True) — so the toggle was silently ignored
    in opposite directions. This reads the real field and additionally
    requires an API key, since cloud calls authenticate with it (see
    invoker_inator.OLLAMA_API_KEY = config.ollama.api_key); with the toggle on
    but no key, it falls back to local rather than making a doomed call.
    """
    cfg = config or ConfigManager().load_config()
    if not cfg.ollama.toggle_cloud:
        return False
    return bool(cfg.ollama.api_key)


class ConfigManager:
    """Handles loading/saving user application configurations exclusively."""

    def load_config(self) -> UserConfig:
        if not CONFIG_FILE.exists():
            return self.generate_default_config()
        with open(CONFIG_FILE, "r") as f:
            return UserConfig(**json.load(f))

    def save_config(self, config: UserConfig):
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config.model_dump(), f, indent=4)

    def generate_default_config(self) -> UserConfig:
        # Fall back to application defaults natively
        config = UserConfig(
            models=ModelConfig(
                chat_model=DEFAULT_CHAT_MODEL,
                embedding_model=DEFAULT_EMBED_MODEL,
                cloud_model=DEFAULT_CLOUD_MODEL,
            )
        )
        self.save_config(config)
        return config

    def update_model_settings(
        self,
        chat=None,
        embedding=None,
        api_key: Optional[str] = None,
        cloud_model: Optional[str] = None,
    ) -> UserConfig:
        config = self.load_config()
        if chat is not None:
            config.models.chat_model = chat
        if embedding is not None:
            config.models.embedding_model = embedding
        if cloud_model is not None:
            config.models.cloud_model = cloud_model
        if api_key is not None:
            config.ollama.api_key = api_key

        self.save_config(config)
        return config

    def get_manifest_data(self) -> dict:
        """Reads from our offline Master File Source of Truth."""
        if not MANIFEST_FILE.exists():
            return {
                "online_chat_models": [],
                "online_embedding_models": [],
                "cloud_models": {},
                "models_details": {},
            }
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
