"""Minimal local vector index — Objective 2's "vector" retrieval path."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .ollama_client import OllamaClient

logger = logging.getLogger("privgraphai.vector_index")


@dataclass
class _IndexedElement:
    element_id: str
    text: str
    embedding: list[float]


class SimpleVectorIndex:
    def __init__(self, ollama_client: OllamaClient, embedding_model: str = "nomic-embed-text"):
        self.llm = ollama_client
        self.embedding_model = embedding_model
        self._items: list[_IndexedElement] = []

    def __len__(self) -> int:
        return len(self._items)

    def build(self, processed_elements: list[dict]) -> None:
        self._items = []
        for el in processed_elements:
            text = el.get("ai_description") or el.get("content", "")
            if not text.strip():
                continue
            try:
                emb = self.llm.embed(text, self.embedding_model)
            except Exception as exc:
                logger.warning("Embedding failed for %s: %s", el["element_id"], exc)
                continue
            self._items.append(_IndexedElement(el["element_id"], text, emb))
        logger.info("Vector index built with %d elements", len(self._items))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self._items:
            return []
        try:
            q_emb = self.llm.embed(query, self.embedding_model)
        except Exception as exc:
            logger.warning("Query embedding failed: %s", exc)
            return []

        scored = [(self._cosine(q_emb, item.embedding), item) for item in self._items]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"element_id": item.element_id, "content": item.text, "score": score}
            for score, item in scored[:top_k]
        ]

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
