import json
import re
import subprocess

import requests
from bs4 import BeautifulSoup

from cerebrum_core.constants import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from cerebrum_core.model_inator import ModelConfig, UserConfig
from cerebrum_core.utils.file_util_inator import CerebrumPaths

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://127.0.0.1:11434"
LIBRARY_URL = "https://ollama.com/library"

CONFIG_ROOT = CerebrumPaths().config_root_dir()
CONFIG_FILE = CONFIG_ROOT / "user_config.json"

EMBED_PATTERN = re.compile(r"(embed|embedding)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# Unified Config Manager
# ─────────────────────────────────────────────────────────────
class ConfigManager:
    """
    Handles:
    - loading/saving user config
    - generating default config
    - detecting Ollama
    - fetching installed & online models
    - pulling models programmatically
    """

    # ─────────────────────────────────────────────────────────
    #  BASIC FILE OPERATIONS
    # ─────────────────────────────────────────────────────────
    def load_config(self) -> UserConfig:
        """Load config or generate defaults if missing."""
        if not CONFIG_FILE.exists():
            return self.generate_default_config()

        with open(CONFIG_FILE, "r") as f:
            return UserConfig(**json.load(f))

    def save_config(self, config: UserConfig):
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config.model_dump(), f, indent=4)

    # ─────────────────────────────────────────────────────────
    #  DEFAULT CONFIG
    # ─────────────────────────────────────────────────────────
    def generate_default_config(self):
        chat, emb = self.get_installed_models()

        # fallback if no models installed yet
        chat = chat or [DEFAULT_CHAT_MODEL]
        emb = emb or [DEFAULT_EMBED_MODEL]

        config = UserConfig(
            models=ModelConfig(
                chat_model=chat[0],
                embedding_model=emb[0],
            )
        )

        self.save_config(config)
        return config

    # ─────────────────────────────────────────────────────────
    #  OLLAMA SYSTEM CHECKS
    # ─────────────────────────────────────────────────────────
    def is_ollama_installed(self):
        try:
            subprocess.run(["ollama", "--version"], stdout=subprocess.PIPE, check=True)
            return True
        except FileNotFoundError:
            return False

    def is_ollama_running(self):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/version", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    def get_ollama_status(self):
        installed = self.is_ollama_installed()
        running = self.is_ollama_running()

        return {
            "installed": installed,
            "running": running,
            "message": (
                "Ollama is ready"
                if installed and running
                else "Ollama is not installed or not running"
            ),
            "install_url": "https://ollama.com/download",
        }

    # ─────────────────────────────────────────────────────────
    #  MODEL OPERATIONS
    # ─────────────────────────────────────────────────────────
    def get_installed_models(self):
        """Return installed models split into chat + embedding."""
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        except Exception:
            return [], []  # if service unavailable

        chat_models = []
        emb_models = []

        for m in response.json().get("models", []):
            name = m.get("name", "")
            (emb_models if EMBED_PATTERN.search(name) else chat_models).append(name)

        return chat_models, emb_models

    def get_available_online_models(self):
        """Fetch full model list available on Ollama.com"""
        response = requests.get(LIBRARY_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        models = set()
        # Fix: Use 'a' tag and get href as string
        for link in soup.find_all("a", href=True):
            href = str(link["href"])  # Convert to string explicitly
            if href.startswith("/library/"):
                model = href.split("/library/")[-1]
                if model:  # Skip empty strings
                    models.add(model)

        online_chat = []
        online_embed = []

        for m in sorted(models):  # Sort for consistency
            (online_embed if EMBED_PATTERN.search(m) else online_chat).append(m)

        return {
            "online_chat_models": sorted(online_chat),
            "online_embedding_models": sorted(online_embed),
        }

    def download_model(self, model_name: str):
        return subprocess.run(["ollama", "pull", model_name], check=False)

    def get_model_details(self, model_name: str):
        """
        Fetch model details from Ollama library page.
        Returns description and available tags/versions with sizes.
        """
        try:
            url = f"https://ollama.com/library/{model_name}/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract description from the main page
            desc_url = f"https://ollama.com/library/{model_name}"
            desc_response = requests.get(desc_url, timeout=10)
            desc_soup = BeautifulSoup(desc_response.text, "html.parser")

            description = None
            desc_elem = desc_soup.find("p", class_="mb-4")
            if desc_elem:
                description = desc_elem.get_text(strip=True)

            # Extract tags with details from the tags table
            tags = []

            # Find all row groups (each tag is in a div with class "group px-4 py-3")
            tag_rows = soup.find_all("div", class_="group px-4 py-3")

            for row in tag_rows:
                tag_info = {}

                # Extract tag name from the link
                tag_link = row.find("a", class_="group-hover:underline")
                if tag_link:
                    tag_text = tag_link.get_text(strip=True)
                    # Extract just the tag part (after the colon)
                    if ":" in tag_text:
                        tag_info["name"] = tag_text.split(":")[-1]
                    else:
                        tag_info["name"] = tag_text

                # Check if this is the latest tag - look for the badge with specific classes
                latest_badges = row.find_all("span")
                for badge in latest_badges:
                    if badge.get_text(strip=True) == "latest":
                        tag_info["is_latest"] = True
                        break

                # Find the grid with size, context, input (desktop view)
                grid = row.find("div", class_="grid grid-cols-12")
                if grid:
                    # Get all <p> tags which contain size, context, input
                    info_cells = grid.find_all("p", class_="text-neutral-500")

                    if len(info_cells) >= 3:
                        size_text = info_cells[0].get_text(strip=True)
                        context_text = info_cells[1].get_text(strip=True)
                        input_text = info_cells[2].get_text(strip=True)

                        if size_text and size_text != "-":
                            tag_info["size"] = size_text
                        if context_text and context_text != "-":
                            tag_info["context"] = context_text
                        if input_text and input_text != "-":
                            tag_info["input"] = input_text

                # Extract digest and date from the bottom line
                digest_line = row.find("span", class_="font-mono")
                if digest_line:
                    tag_info["digest"] = digest_line.get_text(strip=True)

                # Build details string
                details_parts = []
                if tag_info.get("size"):
                    details_parts.append(tag_info["size"])
                if tag_info.get("context"):
                    details_parts.append(f"{tag_info['context']} context")
                if tag_info.get("input"):
                    details_parts.append(f"{tag_info['input']} input")

                if details_parts:
                    tag_info["details"] = " • ".join(details_parts)
                else:
                    tag_info["details"] = ""

                if tag_info.get("name"):
                    tags.append(tag_info)

            # Fallback: if no tags found, provide default
            if not tags:
                tags = [
                    {"name": "latest", "size": "", "details": "", "is_latest": True}
                ]

            return {
                "model_name": model_name,
                "description": description or f"Ollama model: {model_name}",
                "tags": tags,
                "url": url,
            }

        except Exception as e:
            # Return minimal info if scraping fails
            return {
                "model_name": model_name,
                "description": f"Ollama model: {model_name}",
                "tags": [
                    {"name": "latest", "size": "", "details": "", "is_latest": True}
                ],
                "url": f"https://ollama.com/library/{model_name}",
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────
    #  UPDATE USER SETTINGS
    # ─────────────────────────────────────────────────────────
    def update_model_settings(self, chat=None, embedding=None):
        config = self.load_config()

        if chat is not None:
            config.models.chat_model = chat

        if embedding is not None:
            config.models.embedding_model = embedding

        self.save_config(config)
        return config
