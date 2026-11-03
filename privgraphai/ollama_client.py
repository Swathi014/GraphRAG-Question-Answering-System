"""Thin client for a locally running Ollama server.

PROPOSED version. Changes from the original (all optional / backwards compatible):
  * embed_many(): embeds several texts per HTTP call (Ollama's /api/embed accepts a list). The
    original embedded one element per call, which made building a 470-element index take many minutes.
  * generate_text(..., num_predict=N): optional cap on generated tokens, so one runaway generation
    cannot hold the pipeline until the 7-minute timeout. The extraction agent uses it automatically
    when it is available.
  * keep_alive (default "30m"): asks Ollama to keep the model loaded between calls instead of
    unloading it after 5 minutes.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

logger = logging.getLogger("privgraphai.ollama_client")


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 420, keep_alive: str | None = "30m"):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _with_keep_alive(self, payload: dict) -> dict:
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    def generate_text(
        self,
        prompt: str,
        model: str,
        system: str | None = None,
        temperature: float = 0.1,
        num_predict: int | None = None,
    ) -> str:
        options: dict = {"temperature": temperature}
        if num_predict:
            options["num_predict"] = int(num_predict)
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        logger.debug("generate_text -> model=%s prompt_chars=%d", model, len(prompt))
        resp = requests.post(
            f"{self.host}/api/generate", json=self._with_keep_alive(payload), timeout=self.timeout
        )
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
        resp = requests.post(
            f"{self.host}/api/generate", json=self._with_keep_alive(payload), timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def embed(self, text: str, model: str) -> list[float]:
        payload = {"model": model, "input": text}
        logger.debug("embed -> model=%s text_chars=%d", model, len(text))
        resp = requests.post(
            f"{self.host}/api/embed", json=self._with_keep_alive(payload), timeout=self.timeout
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            raise ValueError(f"Ollama /api/embed returned no embeddings for model={model!r}")
        return embeddings[0]

    def embed_many(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed several texts in ONE request. Returns one vector per text, in order."""
        if not texts:
            return []
        payload = {"model": model, "input": list(texts)}
        logger.debug("embed_many -> model=%s texts=%d", model, len(texts))
        resp = requests.post(
            f"{self.host}/api/embed", json=self._with_keep_alive(payload), timeout=self.timeout
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama /api/embed returned {len(embeddings)} embeddings for {len(texts)} texts (model={model!r})"
            )
        return embeddings
