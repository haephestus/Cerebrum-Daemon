import json
import re
import subprocess

import requests
from bs4 import BeautifulSoup

from cerebrum_core.utils.file_util_inator import CerebrumPaths

# ─────────────────────────────────────────────────────────────
# Constants & Envs
# ─────────────────────────────────────────────────────────────
OLLAMA_LOCAL = "http://127.0.0.1:11434"
LIBRARY_URL = "https://ollama.com/library"
MANIFEST_FILE = CerebrumPaths().config_root_dir() / "models_manifest.json"

EMBED_PATTERN = re.compile(r"(embed|embedding)", re.IGNORECASE)
CLOUD_PATTERN = re.compile(r"[-_]cloud", re.IGNORECASE)


class OllamaManifestGenerator:
    """Scrapes Ollama once, processes strings, and updates our static Master JSON File."""

    # ─────────────────────────────────────────────────────────
    #  OLLAMA SYSTEM CHECKS
    # ─────────────────────────────────────────────────────────
    def is_ollama_installed(self) -> bool:
        try:
            subprocess.run(["ollama", "--version"], stdout=subprocess.PIPE, check=True)
            return True
        except FileNotFoundError:
            return False

    def is_ollama_running(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_LOCAL}/api/version", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    def get_ollama_status(self) -> dict:
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
    #  LOCAL CLI AND INVENTORY
    # ─────────────────────────────────────────────────────────
    def get_installed_models(self):
        """Return installed models split into chat + embedding using local api."""
        try:
            response = requests.get(f"{OLLAMA_LOCAL}/api/tags")
            response.raise_for_status()
        except Exception:
            return [], []  # if service unavailable

        chat_models = []
        emb_models = []

        for m in response.json().get("models", []):
            name = m.get("name", "")
            (emb_models if EMBED_PATTERN.search(name) else chat_models).append(name)

        return chat_models, emb_models

    def download_model(self, model_name: str):
        return subprocess.run(["ollama", "pull", model_name], check=False)

    # ─────────────────────────────────────────────────────────
    #  ONLINE WEB PARSING & PROCESSING
    # ─────────────────────────────────────────────────────────
    def fetch_all_available_models(self) -> list[str]:
        """Gets all base model names from the main library page."""
        try:
            response = requests.get(LIBRARY_URL, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            models = set()
            for link in soup.find_all("a", href=True):
                href = str(link["href"])
                if href.startswith("/library/"):
                    model = href.split("/library/")[-1]
                    if model:
                        models.add(model)
            return sorted(list(models))
        except Exception as e:
            print(f"Error fetching base models: {e}")
            return []

    def get_model_details(self, model_name: str) -> dict:
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

            tags = []
            tag_rows = soup.find_all("div", class_="group px-4 py-3")

            for row in tag_rows:
                tag_info = {}

                # Extract tag name from the link
                tag_link = row.find("a", class_="group-hover:underline")
                if tag_link:
                    tag_text = tag_link.get_text(strip=True)
                    if ":" in tag_text:
                        tag_info["name"] = tag_text.split(":")[-1]
                    else:
                        tag_info["name"] = tag_text

                # Check if this is the latest tag
                latest_badges = row.find_all("span")
                for badge in latest_badges:
                    if badge.get_text(strip=True) == "latest":
                        tag_info["is_latest"] = True
                        break

                # Find the grid with size, context, input
                grid = row.find("div", class_="grid grid-cols-12")
                if grid:
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

                tag_info["details"] = " • ".join(details_parts) if details_parts else ""

                if tag_info.get("name"):
                    tags.append(tag_info)

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
            return {
                "model_name": model_name,
                "description": f"Ollama model: {model_name}",
                "tags": [
                    {"name": "latest", "size": "", "details": "", "is_latest": True}
                ],
                "url": f"https://ollama.com/library/{model_name}",
                "error": str(e),
            }

    def filter_cloud_tags_from_details(self, model_details: dict) -> list[str]:
        """Extracts only tag names matching your cloud pattern locally."""
        return [
            tag["name"]
            for tag in model_details.get("tags", [])
            if CLOUD_PATTERN.search(tag.get("name", ""))
        ]

    # ─────────────────────────────────────────────────────────
    #  MANIFEST ENGINE BUILDER
    # ─────────────────────────────────────────────────────────
    def build_master_manifest(self):
        """Orchestrates compilation and updates the master JSON file."""
        print("Starting Ollama library sync...")
        base_models = self.fetch_all_available_models()

        online_chat = []
        online_embed = []
        cloud_map = {}
        details_map = {}

        for model in base_models:
            # Segregate base names
            if EMBED_PATTERN.search(model):
                online_embed.append(model)
            else:
                online_chat.append(model)

            # Deep scrape details and isolate clouds
            print(f"Parsing details for: {model}")
            details = self.get_model_details(model)
            details_map[model] = details

            # Isolate strings with cloud markers out of the current tags array iteration loop
            cloud_tags = self.filter_cloud_tags_from_details(details)
            if cloud_tags:
                cloud_map[model] = cloud_tags

        manifest = {
            "online_chat_models": online_chat,
            "online_embedding_models": online_embed,
            "cloud_models": cloud_map,
            "models_details": details_map,
        }

        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=4)
        print("Master manifest compiled successfully!")
        return manifest


if __name__ == "__main__":
    # Safely execute the sequence out of structural runtime limits
    generator = OllamaManifestGenerator()
    generator.build_master_manifest()
