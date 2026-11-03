"""Minimal local vector index - Objective 2's "vector" retrieval path.

PROPOSED version. What changed and why:
  * build() embeds in batches (embed_many) instead of one HTTP call per element.
  * The embeddings can be saved to disk and reloaded (save / load / build_or_load). Previously every
    `ask` process re-embedded every element before it could answer, which took many minutes per
    question in the web UI. build_or_load() reuses a saved index if it matches the elements and the
    embedding model, and otherwise builds it once and saves it for next time.
  * Vectors are normalised once, so a search is one dot product per element.
Public methods (build, search, __len__) and the result format are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ollama_client import OllamaClient

logger = logging.getLogger("privgraphai.vector_index")

_BATCH_SIZE = 16


@dataclass
class _IndexedElement:
    element_id: str
    text: str
    embedding: list[float]  # stored normalised to unit length


def _element_text(el: dict) -> str:
    return el.get("ai_description") or el.get("content", "")


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)


class SimpleVectorIndex:
    def __init__(self, ollama_client: OllamaClient, embedding_model: str = "nomic-embed-text"):
        self.llm = ollama_client
        self.embedding_model = embedding_model
        self._items: list[_IndexedElement] = []

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ build
    def build(self, processed_elements: list[dict]) -> None:
        self._items = []
        pending = [(el["element_id"], _element_text(el)) for el in processed_elements]
        pending = [(eid, text) for eid, text in pending if text.strip()]

        for start in range(0, len(pending), _BATCH_SIZE):
            batch = pending[start : start + _BATCH_SIZE]
            for (eid, text), emb in zip(batch, self._embed_batch([t for _, t in batch], [e for e, _ in batch])):
                if emb is not None:
                    self._items.append(_IndexedElement(eid, text, _normalise(emb)))
        logger.info("Vector index built with %d elements", len(self._items))

    def _embed_batch(self, texts: list[str], ids: list[str]) -> list[Optional[list[float]]]:
        """One request per batch; if it fails, fall back to one request per text."""
        embed_many = getattr(self.llm, "embed_many", None)
        if embed_many is not None:
            try:
                return list(embed_many(texts, self.embedding_model))
            except Exception as exc:
                logger.warning("Batch embedding failed (%s); retrying one by one", exc)
        out: list[Optional[list[float]]] = []
        for eid, text in zip(ids, texts):
            try:
                out.append(self.llm.embed(text, self.embedding_model))
            except Exception as exc:
                logger.warning("Embedding failed for %s: %s", eid, exc)
                out.append(None)
        return out

    # ------------------------------------------------------------ save / load
    def _fingerprint(self, processed_elements: list[dict]) -> str:
        h = hashlib.sha1(self.embedding_model.encode("utf-8"))
        for el in processed_elements:
            h.update(b"\x00" + str(el["element_id"]).encode("utf-8") + b"\x01" + _element_text(el).encode("utf-8"))
        return h.hexdigest()

    def save(self, path: str | Path, fingerprint: str = "") -> None:
        data = {
            "model": self.embedding_model,
            "fingerprint": fingerprint,
            "items": [
                {"element_id": i.element_id, "text": i.text, "embedding": i.embedding} for i in self._items
            ],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        logger.info("Vector index saved to %s (%d elements)", path, len(self._items))

    def load(self, path: str | Path) -> str:
        """Load a saved index; returns the fingerprint it was saved with."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._items = [
            _IndexedElement(i["element_id"], i["text"], _normalise(i["embedding"])) for i in data["items"]
        ]
        logger.info("Vector index loaded from %s (%d elements)", path, len(self._items))
        return data.get("fingerprint", "")

    def build_or_load(self, processed_elements: list[dict], cache_path: str | Path) -> None:
        """Reuse the saved index if it matches these elements and this model; else build and save it."""
        fingerprint = self._fingerprint(processed_elements)
        cache = Path(cache_path)
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data.get("model") == self.embedding_model and data.get("fingerprint") == fingerprint:
                    self.load(cache)
                    return
                logger.info("Saved vector index does not match the elements/model; rebuilding.")
            except Exception as exc:
                logger.warning("Could not read saved vector index (%s); rebuilding.", exc)
        self.build(processed_elements)
        if len(self._items):
            try:
                self.save(cache, fingerprint)
            except OSError as exc:
                logger.warning("Could not save vector index to %s: %s", cache, exc)

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self._items:
            return []
        try:
            q_emb = _normalise(self.llm.embed(query, self.embedding_model))
        except Exception as exc:
            logger.warning("Query embedding failed: %s", exc)
            return []

        scored = [(self._dot(q_emb, item.embedding), item) for item in self._items]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"element_id": item.element_id, "content": item.text, "score": score}
            for score, item in scored[:top_k]
        ]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
