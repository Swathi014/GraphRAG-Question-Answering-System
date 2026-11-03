"""
Knowledge-graph construction agent - Objective 2.

Takes the `processed_elements` list produced by Objective 1's ingestion
pipeline and, for each element, prompts the local LLM to extract named
entities and the relations between them, then writes both into a
`KnowledgeGraph`.

Real-world runs against a full research paper (700+ elements) surfaced two
failure modes small quantized models produce under load, which the parsing
below specifically handles:
  1. Concatenated JSON: the model emits more than one `{...}` blob in a
     single response (e.g. an empty `{}` "false start" followed by the real
     answer on the next line).
  2. Truncated/malformed JSON: the model forgets to close a bracket, e.g.
     `..."target": "surged"}}}` where the `relations` array was never closed
     with `]`.

PROPOSED version. Further hardening found in later runs:
  3. Valid JSON with the wrong TYPES (a list where a name should be, entries that are strings
     instead of objects). Entities and relations are now sanitised before they reach the graph, and
     anything unusable is skipped and counted instead of crashing the whole ingestion.
  4. One bad element can no longer stop ingestion: linking is wrapped per item.
  5. Reference-list entries (long author lists with years) are not sent to the model. They made it
     run on until the 7-minute timeout and add nothing to question answering. The element itself
     stays in the graph, so it can still be retrieved.
  6. If the Ollama client's generate_text accepts a token cap (parameter `num_predict` or
     `max_tokens`), extraction uses one, so a runaway generation stops early.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
from typing import Optional

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

_SKIP_REFERENCE_LIKE = True
_MAX_NEW_TOKENS = 700  # used only if the client supports a token cap

_CITATION_START = re.compile(r"^\s*\[\d{1,3}\]\s")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_VENUE_WORDS = re.compile(r"(arXiv|Proceedings|Conference|Journal|doi|https?://|Trans\.|Workshop)", re.I)


def _is_reference_like(text: str) -> bool:
    """True for bibliography entries: '[12] ...' or a long, comma-heavy line with a year and a venue."""
    if _CITATION_START.match(text):
        return True
    return (
        len(text) > 200
        and text.count(",") >= 6
        and bool(_YEAR.search(text))
        and bool(_VENUE_WORDS.search(text))
    )


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str):
        value = " ".join(value.split())
        return value or None
    return None


def _sanitize_entities(entities) -> list[dict]:
    """Keep only {name: str, type: str} entries; drop anything else."""
    if not isinstance(entities, list):
        return []
    clean = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = _clean_str(ent.get("name"))
        if name:
            clean.append({"name": name, "type": _clean_str(ent.get("type")) or "unknown"})
    return clean


def _sanitize_relations(relations) -> list[dict]:
    """Keep only relations whose source, relation and target are all non-empty strings."""
    if not isinstance(relations, list):
        return []
    clean = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        source, label, target = (_clean_str(rel.get(k)) for k in ("source", "relation", "target"))
        if source and label and target:
            clean.append({"source": source, "relation": label, "target": target})
    return clean


class ExtractionAgent:
    """Builds a KnowledgeGraph from a list of processed element dicts (Objective 1's output)."""

    def __init__(self, config: PrivGraphAIConfig, ollama_client: OllamaClient):
        self.config = config
        self.llm = ollama_client

    def build_graph(
        self,
        processed_elements: list[dict],
        stop_event: Optional[threading.Event] = None,
        progress_cb=None,
    ) -> KnowledgeGraph:
        """
        `stop_event`: if provided and set between elements, extraction stops
        early and returns the graph built so far (cooperative cancellation -
        checked between elements, not mid-call, since an in-flight Ollama
        request can't be interrupted without closing the connection).
        `progress_cb`: optional callable(done: int, total: int) invoked after
        each element, for a caller that wants to show live progress.
        """
        kg = KnowledgeGraph()

        for el in processed_elements:
            kg.add_element(
                element_id=el["element_id"],
                element_type=el["element_type"],
                content=el.get("ai_description") or el.get("content", ""),
                page_number=el.get("page_number"),
            )

        total = len(processed_elements)
        skipped_refs = 0
        bad_items = 0
        for i, el in enumerate(processed_elements):
            if stop_event is not None and stop_event.is_set():
                logger.info("Extraction stopped early at element %d/%d (by request).", i, total)
                break

            text = el.get("ai_description") or el.get("content", "")
            if _SKIP_REFERENCE_LIKE and _is_reference_like(text):
                skipped_refs += 1
            elif len(text.strip()) >= self.config.min_extraction_chars:
                try:
                    entities, relations = self._extract_one(text)
                except Exception as exc:
                    logger.warning("Extraction failed for %s: %s", el["element_id"], exc)
                    entities, relations = [], []

                for ent in entities:
                    try:
                        kg.link_entity_to_element(ent["name"], el["element_id"], ent["type"])
                    except Exception as exc:  # one bad entity must not stop the ingestion
                        bad_items += 1
                        logger.debug("Skipping entity in %s: %s", el["element_id"], exc)
                for rel in relations:
                    try:
                        kg.add_relation(rel["source"], rel["relation"], rel["target"])
                    except Exception as exc:
                        bad_items += 1
                        logger.debug("Skipping relation in %s: %s", el["element_id"], exc)

            if progress_cb is not None:
                try:
                    progress_cb(i + 1, total)
                except Exception:
                    pass  # progress reporting must never break extraction

        logger.info(
            "Knowledge graph built: %s (reference-like elements skipped: %d, unusable items skipped: %d)",
            kg.stats(), skipped_refs, bad_items,
        )
        return kg

    def _extract_one(self, content: str) -> tuple[list[dict], list[dict]]:
        # Capped at 1200 chars: a dense citation list or reference block can
        # otherwise push the model into enumerating dozens of entities,
        # which is what drove some calls past the timeout in testing on a
        # full research paper.
        prompt = _EXTRACTION_PROMPT.format(content=content[:1200])
        raw = self._generate(prompt)
        parsed = self._parse_json_relaxed(raw)
        return (
            _sanitize_entities(parsed.get("entities", [])),
            _sanitize_relations(parsed.get("relations", [])),
        )

    def _generate(self, prompt: str) -> str:
        """Call the model; add a token cap only if the client supports one."""
        kwargs = {"temperature": 0.0}
        try:
            params = inspect.signature(self.llm.generate_text).parameters
            for cap_name in ("num_predict", "max_tokens"):
                if cap_name in params:
                    kwargs[cap_name] = _MAX_NEW_TOKENS
                    break
        except (TypeError, ValueError):
            pass
        return self.llm.generate_text(prompt, self.config.text_model, **kwargs)

    @staticmethod
    def _parse_json_relaxed(raw: str) -> dict:
        """Parse the model's JSON output, tolerating two failure modes
        observed in real runs: multiple concatenated objects, and a single
        object with an unclosed bracket. Falls back to an empty result
        (never raises) if nothing usable can be recovered."""
        raw = raw.strip()
        if not raw:
            return {"entities": [], "relations": []}

        candidates = ExtractionAgent._find_balanced_json_objects(raw)

        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("entities" in obj or "relations" in obj):
                return obj

        target = candidates[-1] if candidates else raw[raw.find("{"):] if "{" in raw else raw
        repaired = ExtractionAgent._try_repair_json(target)
        if repaired is not None:
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM JSON output: %.300s", raw)
        return {"entities": [], "relations": []}

    @staticmethod
    def _find_balanced_json_objects(raw: str) -> list[str]:
        objects = []
        depth = 0
        start: Optional[int] = None
        in_string = False
        escape = False
        for i, ch in enumerate(raw):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objects.append(raw[start : i + 1])
                        start = None
        return objects

    @staticmethod
    def _try_repair_json(raw: str) -> Optional[str]:
        stack: list[str] = []
        in_string = False
        escape = False
        keep: list[str] = []
        changed = False
        closers = {"{": "}", "[": "]"}

        for ch in raw:
            if in_string:
                keep.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                keep.append(ch)
            elif ch in "{[":
                stack.append(ch)
                keep.append(ch)
            elif ch in "}]":
                if stack and closers[stack[-1]] == ch:
                    stack.pop()
                    keep.append(ch)
                else:
                    changed = True
            else:
                keep.append(ch)

        if stack:
            changed = True
            keep.extend(closers[c] for c in reversed(stack))

        return "".join(keep) if changed else None
