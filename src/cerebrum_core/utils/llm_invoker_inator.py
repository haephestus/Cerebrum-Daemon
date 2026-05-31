import json
import logging

import requests

from cerebrum_core.user_inator import ConfigManager


def ollama_structured(prompt: str, analyses_schema: dict) -> str:
    config = ConfigManager().load_config()
    base_url = getattr(config.models, "ollama_base_url", "http://127.0.0.1:11434")

    payload = {
        "model": config.models.chat_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 32768},
        "format": analyses_schema,
    }

    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()

    raw = resp.json()
    logging.info(f"[OLLAMA] response keys: {list(raw.keys())}")

    response_text = raw.get("response")

    if not response_text:
        logging.error(
            f"[OLLAMA] empty/null response field. Full payload: {json.dumps(raw)[:1000]}"
        )
        raise ValueError(
            f"Ollama returned no response content. Keys: {list(raw.keys())}"
        )

    logging.info(f"[OLLAMA] response received — {len(response_text)} chars")
    return response_text
