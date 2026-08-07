import json
import logging
from typing import Any

import requests
from ollama import Client

from cerebrum_core.user_inator import ConfigManager

OLLAMA_CLOUD = "https://ollama.com"
OLLAMA_API_KEY = ConfigManager().load_config().ollama.api_key
is_local = ConfigManager().load_config().ollama.toggle_cloud


def ollama_response(is_local: bool, schema: dict, prompt: str) -> dict[str, Any]:
    """
    Dynamic routing to ollama local or ollama cloud
    """
    result = {}
    if is_local:
        result = ollama_local_call2(prompt=prompt, schema=schema)
    else:
        result = ollama_cloud_call(prompt=prompt, schema=schema)
    return result


def ollama_local_call(prompt: str, analyses_schema: dict) -> str:
    config = ConfigManager().load_config()
    base_url = getattr(config.models, "ollama_base_url", "http://127.0.0.1:11434")

    payload = {
        "model": config.models.chat_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
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


def ollama_local_call2(
    prompt: str, schema: dict, system_prompt: str = ""
) -> dict[str, Any]:
    config = ConfigManager().load_config()
    base_url = getattr(config.models, "ollama_base_url", "http://127.0.0.1:11434")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": config.models.chat_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
        "format": schema,
    }

    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    raw = resp.json()

    logging.info(f"[OLLAMA] response keys: {list(raw.keys())}")
    logging.info(f"[OLLAMA] done_reason: {raw.get('done_reason')}")

    response_text = raw.get("message", {}).get("content")

    if not response_text:
        logging.error(
            f"[OLLAMA] empty/null response field. Full payload: {json.dumps(raw)[:1000]}"
        )
        raise ValueError(
            f"Ollama returned no response content. Keys: {list(raw.keys())}, "
            f"done_reason: {raw.get('done_reason')}"
        )

    logging.info(f"[OLLAMA] response received — {len(response_text)} chars")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError("Model returned invalid JSON.") from e


def ollama_cloud_call(
    prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    config = ConfigManager().load_config()
    model = str(getattr(config.models, "cloud_chat_model", config.models.cloud_model))

    client = Client(
        host=OLLAMA_CLOUD,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )

    response = client.generate(
        model=model,
        prompt=prompt,
        format=schema,
        options={"temperature": 0},
    )

    if not response.response:
        raise ValueError("Ollama cloud returned no response content.")

    try:
        return json.loads(response.response)
    except json.JSONDecodeError as e:
        raise ValueError("Model returned invalid JSON.") from e
