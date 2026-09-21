"""
Routing agent — Objective 2.

Given a question, decides which retrieval strategy PrivGraphAI should use:
  - "graph"  : the knowledge-graph, multi-hop walk (comparisons, "how does
               X relate to Y", questions needing several connected facts)
  - "vector" : direct embedding similarity lookup (single-fact questions)
  - "hybrid" : both, merged

Tries an LLM classification first; falls back to a small keyword heuristic
if the model's output can't be parsed, so this step never blocks the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Optional

from .config import PrivGraphAIConfig
from .ollama_client import OllamaClient

logger = logging.getLogger("privgraphai.routing_agent")

RetrievalStrategy = Literal["graph", "vector", "hybrid"]
_VALID_STRATEGIES = ("graph", "vector", "hybrid")

_ROUTING_PROMPT = """Classify how to best answer the QUESTION below using a document knowledge base.

Return ONLY a JSON object like {{"strategy": "graph" | "vector" | "hybrid", "reason": "..."}}.

- "graph": the question needs connecting facts across multiple entities/parts of the document (multi-hop, comparisons, "how does X relate to Y").
- "vector": the question is a direct, single-fact lookup answerable from one passage.
- "hybrid": you are unsure, or the question could benefit from both.

QUESTION: {question}
"""

_MULTI_HOP_MARKERS = (
    "compare", "comparison", "relationship", "between", "versus", " vs ",
    "how does", "why does", "difference", "related to", "affects",
)


class RoutingAgent:
    def __init__(self, config: PrivGraphAIConfig, ollama_client: OllamaClient):
        self.config = config
        self.llm = ollama_client

    def route(self, question: str) -> RetrievalStrategy:
        try:
            raw = self.llm.generate_text(
                _ROUTING_PROMPT.format(question=question), self.config.text_model, temperature=0.0
            )
            strategy = self._parse(raw)
            if strategy:
                logger.info("Routed question to strategy=%s", strategy)
                return strategy
        except Exception as exc:
            logger.warning("Routing LLM call failed, falling back to heuristic: %s", exc)

        fallback = self._heuristic_fallback(question)
        logger.info("Routed question to strategy=%s (heuristic fallback)", fallback)
        return fallback

    @staticmethod
    def _parse(raw: str) -> Optional[RetrievalStrategy]:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        strategy = obj.get("strategy")
        return strategy if strategy in _VALID_STRATEGIES else None

    @staticmethod
    def _heuristic_fallback(question: str) -> RetrievalStrategy:
        q_lower = question.lower()
        if any(marker in q_lower for marker in _MULTI_HOP_MARKERS):
            return "hybrid"
        return "vector"
