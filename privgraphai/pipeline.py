"""PrivGraphAI on-device ingestion pipeline - Objective 1.

PROPOSED version. Changes from the original:
  * Figures: the indexed description now combines the caption, the text found inside
    the figure (grouped in columns by the parser) and the Moondream description, so
    chart values survive even if the vision model misreads them. A failing vision call
    no longer drops the whole figure.
  * Formulas: no LLM paraphrase (a 3B model can change signs and symbols). The LaTeX
    is kept as is, plus the sentence that follows it when it starts with "where"/"with"
    (it defines the symbols). Other neighbouring text is not added, because it made the
    formula element match unrelated questions.
  * Tables: the LLM description is kept, and the exact markdown table is appended so
    numbers are never lost to a paraphrase. Row elements and table overviews (large tables are split
    into one element per row by the parser) are used as they are: no LLM call, so a 30-row table does
    not cost 30 paraphrase calls.
"""

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
    "shows and any labeled values. Only state numbers that are clearly legible. "
    "Do not speculate beyond what is visible."
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

        for idx, el in enumerate(elements):
            try:
                processed.append(
                    self._process_single_element(el, self._neighbour_text(elements, idx))
                )
            except Exception as exc:
                logger.warning("Skipping element %s: %s", el.element_id, exc)
                errors.append(f"process_elements[{el.element_id}]: {exc}")

        return {"processed_elements": processed, "errors": errors}

    @staticmethod
    def _neighbour_text(elements: list[MultimodalElement], idx: int) -> tuple[str, str]:
        """Text element just before and just after elements[idx] (for formula context)."""
        prev_text = next_text = ""
        if idx > 0 and elements[idx - 1].element_type == "text":
            prev_text = elements[idx - 1].content.strip()
        if idx + 1 < len(elements) and elements[idx + 1].element_type == "text":
            next_text = elements[idx + 1].content.strip()
        return prev_text, next_text

    def _process_single_element(
        self, el: MultimodalElement, context: tuple[str, str] = ("", "")
    ) -> dict:
        record = asdict(el)

        if el.element_type == "figure":
            parts: list[str] = []
            if el.content.strip():
                parts.append(f"Figure caption: {el.content.strip()}")
            figure_text = (el.metadata or {}).get("figure_text", "")
            if figure_text:
                parts.append(f"Text found inside the figure. {figure_text}")
            if el.image_path:
                try:
                    visual = self.llm.describe_image(
                        el.image_path, _FIGURE_CAPTION_PROMPT, self.config.vision_model
                    )
                    if visual and visual.strip():
                        parts.append(f"Visual description: {visual.strip()}")
                except Exception as exc:  # keep the figure even if the vision model fails
                    logger.warning("Vision description failed for %s: %s", el.element_id, exc)
            record["ai_description"] = "\n".join(parts) if parts else None

        elif el.element_type == "formula":
            if el.content.strip():
                _, next_text = context
                parts = [f"Formula: {el.content.strip()}"]
                if next_text.lower().startswith(("where", "with", "in which")):
                    parts.append(next_text)  # the sentence that defines the symbols
                record["ai_description"] = "\n".join(parts)
            else:
                record["ai_description"] = None

        elif (
            el.element_type == "table"
            and el.content.strip()
            and not (el.metadata or {}).get("table_row")
            and not (el.metadata or {}).get("table_overview")
        ):
            prompt = _TEXT_NORMALIZE_PROMPT.format(
                kind=el.element_type, kind_upper=el.element_type.upper(), content=el.content
            )
            description = self.llm.generate_text(prompt, self.config.text_model)
            record["ai_description"] = f"{description}\n\nTable (markdown):\n{el.content}"

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
