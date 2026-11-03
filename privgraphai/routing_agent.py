"""Routing agent - Objective 2.

PROPOSED version. Only the prompt changed: it now contains six generic examples (none taken from
the evaluation questions) and no longer asks for a "reason", which saves generated tokens.
On the 29-question router evaluation the few-shot prompt agreed with the labels on 19 of 29
questions against 15 of 29 for the original prompt, and ran about 35% faster.
The parsing and the keyword fallback are unchanged.
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

_ROUTING_PROMPT = """Decide how to retrieve evidence for the QUESTION from a document knowledge base.

Return ONLY a JSON object: {{"strategy": "vector"}} or {{"strategy": "graph"}} or {{"strategy": "hybrid"}}.

- "vector": ONE fact, number, name or definition that sits in a single passage, table row or figure.
- "graph": the answer needs several facts about DIFFERENT entities connected together (comparisons, lists across the document, "how does X relate to Y").
- "hybrid": it might need both, or you are not sure.

Examples:
QUESTION: Who wrote the introduction? -> {{"strategy": "vector"}}
QUESTION: What melting point does Table 2 give for compound X? -> {{"strategy": "vector"}}
QUESTION: How many samples does the dataset contain? -> {{"strategy": "vector"}}
QUESTION: Compare method A and method B on accuracy and cost. -> {{"strategy": "graph"}}
QUESTION: Which approaches in the report use sensors, and what result does each one report? -> {{"strategy": "graph"}}
QUESTION: Which option performed best, and why do the authors think it worked? -> {{"strategy": "hybrid"}}

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
