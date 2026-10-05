"""PrivGraphAI on-device ingestion pipeline — Objective 1."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from langgraph.graph import END, StateGraph

from .config import DEFAULT_CONFIG, PrivGraphAIConfig
from .docling_parser import DoclingParser, MultimodalElement
from .ollama_client import OllamaClient
from .pipeline_state import PrivGraphAIState

logger = logging.getLogger("privgraphai.pipeline")

_TEXT_NORMALIZE_PROMPT = (
    "Rewrite the following extracted document {kind} as a single, clean, "
    "self-contained statement suitable for indexing. Do not add information "
    "that is not present in it.\n\n{kind_upper}:\n{content}"
)

_FIGURE_CAPTION_PROMPT = (
    "Describe this figure factually in 1-3 sentences, focusing on what it "
    "shows and any labeled values. Do not speculate beyond what is visible."
)


class PrivGraphAIPipeline:
    def __init__(
        self,
        config: PrivGraphAIConfig = DEFAULT_CONFIG,
        docling_parser: Optional[DoclingParser] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.config = config
        self.parser = docling_parser or DoclingParser(config.image_export_dir)
        self.llm = ollama_client or OllamaClient(config.ollama_host, config.request_timeout_s)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PrivGraphAIState)
        graph.add_node("parse_document", self._parse_document_node)
        graph.add_node("process_elements", self._process_elements_node)
        graph.set_entry_point("parse_document")
        graph.add_edge("parse_document", "process_elements")
        graph.add_edge("process_elements", END)
        return graph.compile()

    def _parse_document_node(self, state: PrivGraphAIState) -> dict:
        source = state["source_path"]
        try:
            elements = self.parser.parse(source)
            return {"raw_elements": elements}
        except Exception as exc:
            logger.exception("Docling parsing failed for %s", source)
            return {"raw_elements": [], "errors": [f"parse_document: {exc}"]}

    def _process_elements_node(self, state: PrivGraphAIState) -> dict:
        elements = state.get("raw_elements", [])
        errors = list(state.get("errors", []))
        processed: list[dict] = []

        for el in elements:
            try:
                processed.append(self._process_single_element(el))
            except Exception as exc:
                logger.warning("Skipping element %s: %s", el.element_id, exc)
                errors.append(f"process_elements[{el.element_id}]: {exc}")

        return {"processed_elements": processed, "errors": errors}

    def _process_single_element(self, el: MultimodalElement) -> dict:
        record = asdict(el)

        if el.element_type == "figure" and el.image_path:
            record["ai_description"] = self.llm.describe_image(
                el.image_path, _FIGURE_CAPTION_PROMPT, self.config.vision_model
            )
        elif el.element_type in ("table", "formula") and el.content.strip():
            prompt = _TEXT_NORMALIZE_PROMPT.format(
                kind=el.element_type, kind_upper=el.element_type.upper(), content=el.content
            )
            record["ai_description"] = self.llm.generate_text(prompt, self.config.text_model)
        else:
            record["ai_description"] = None

        return record

    def run(self, source_path: str) -> PrivGraphAIState:
        if not self.llm.is_available():
            logger.warning(
                "Ollama server at %s is not reachable. Is `ollama serve` running, "
                "and have the models been pulled?",
                self.config.ollama_host,
            )
        initial_state: PrivGraphAIState = {"source_path": source_path, "errors": []}
        return self.graph.invoke(initial_state)
