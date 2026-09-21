"""
Knowledge-graph construction agent — Objective 2.

Takes the `processed_elements` list produced by Objective 1's ingestion
pipeline and, for each element, prompts the local LLM to extract named
entities and the relations between them, then writes both into a
`KnowledgeGraph`.
"""

from __future__ import annotations

import json
import logging

from .config import PrivGraphAIConfig
from .graph_store import KnowledgeGraph
from .ollama_client import OllamaClient

logger = logging.getLogger("privgraphai.extraction_agent")

_EXTRACTION_PROMPT = """You extract entities and relations for a knowledge graph.
Read the CONTENT below and return ONLY a JSON object (no prose, no markdown fences) shaped exactly like:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}

Rules:
- Only include entities and relations explicitly present in the content.
- "type" is a short label such as "model", "metric", "method", "organization", "concept".
- If nothing meaningful can be extracted, return {{"entities": [], "relations": []}}.

CONTENT:
{content}
"""


class ExtractionAgent:
    """Builds a KnowledgeGraph from a list of processed element dicts (Objective 1's output)."""

    def __init__(self, config: PrivGraphAIConfig, ollama_client: OllamaClient):
        self.config = config
        self.llm = ollama_client

    def build_graph(self, processed_elements: list[dict]) -> KnowledgeGraph:
        kg = KnowledgeGraph()

        # 1. Register every element as a node first, so entity->element edges
        #    always resolve regardless of extraction order.
        for el in processed_elements:
            kg.add_element(
                element_id=el["element_id"],
                element_type=el["element_type"],
                content=el.get("ai_description") or el.get("content", ""),
                page_number=el.get("page_number"),
            )

        # 2. Extract entities/relations per element and wire them in.
        for el in processed_elements:
            text = el.get("ai_description") or el.get("content", "")
            if not text.strip():
                continue
            try:
                entities, relations = self._extract_one(text)
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", el["element_id"], exc)
                continue

            for ent in entities:
                name = ent.get("name")
                if name:
                    kg.link_entity_to_element(name, el["element_id"], ent.get("type", "unknown"))
            for rel in relations:
                if rel.get("source") and rel.get("relation") and rel.get("target"):
                    kg.add_relation(rel["source"], rel["relation"], rel["target"])

        logger.info("Knowledge graph built: %s", kg.stats())
        return kg

    def _extract_one(self, content: str) -> tuple[list[dict], list[dict]]:
        prompt = _EXTRACTION_PROMPT.format(content=content[:2000])  # keep prompts small-model-friendly
        raw = self.llm.generate_text(prompt, self.config.text_model, temperature=0.0)
        parsed = self._parse_json_relaxed(raw)
        return parsed.get("entities", []), parsed.get("relations", [])

    @staticmethod
    def _parse_json_relaxed(raw: str) -> dict:
        """Small local models sometimes wrap JSON in markdown fences or add
        stray text around it. Extract the first {...} block and parse that."""
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {"entities": [], "relations": []}
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM JSON output: %.200s", raw)
            return {"entities": [], "relations": []}
