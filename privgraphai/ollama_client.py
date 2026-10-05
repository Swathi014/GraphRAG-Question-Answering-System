"""Thin client for a locally running Ollama server."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

logger = logging.getLogger("privgraphai.ollama_client")


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 420):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def generate_text(self, prompt: str, model: str, system: str | None = None, temperature: float = 0.1) -> str:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        logger.debug("generate_text -> model=%s prompt_chars=%d", model, len(prompt))
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def describe_image(self, image_path: str | Path, prompt: str, model: str) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        logger.debug("describe_image -> model=%s image=%s", model, image_path)
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def embed(self, text: str, model: str) -> list[float]:
        payload = {"model": model, "input": text}
        logger.debug("embed -> model=%s text_chars=%d", model, len(text))
        resp = requests.post(f"{self.host}/api/embed", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            raise ValueError(f"Ollama /api/embed returned no embeddings for model={model!r}")
        return embeddings[0]
