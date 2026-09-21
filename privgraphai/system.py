"""
Top-level PrivGraphAI facade — combines Objective 1 (ingestion) and
Objective 2 (knowledge-graph construction + routing/verification QA).

    ingest(pdf_path)  -> Docling + Ollama ingestion (Obj 1), then knowledge-
                         graph construction + vector index build (Obj 2)
    save(path) / load(path, ...) -> persist/restore the knowledge graph so
                         `ingest` and `ask` can run as separate invocations
    ask(question)     -> route -> retrieve -> verify (loop) -> answer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG, PrivGraphAIConfig
from .docling_parser import DoclingParser
from .extraction_agent import ExtractionAgent
from .graph_store import KnowledgeGraph
from .ollama_client import OllamaClient
from .pipeline import PrivGraphAIPipeline
from .qa_pipeline import QAPipeline
from .vector_index import SimpleVectorIndex

logger = logging.getLogger("privgraphai.system")


class PrivGraphAISystem:
    def __init__(
        self,
        config: PrivGraphAIConfig = DEFAULT_CONFIG,
        docling_parser: Optional[DoclingParser] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.config = config
        self.llm = ollama_client or OllamaClient(config.ollama_host, config.request_timeout_s)
        self.ingestion = PrivGraphAIPipeline(config, docling_parser=docling_parser, ollama_client=self.llm)
        self.extraction_agent = ExtractionAgent(config, self.llm)
        self.vector_index = SimpleVectorIndex(self.llm, config.embedding_model)
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self._last_processed_elements: list[dict] = []

    # -- Objective 1 + graph construction (Objective 2, part 1) -----------

    def ingest(self, source_path: str) -> dict:
        """Run ingestion (Objective 1), then build the knowledge graph and
        vector index from its output (Objective 2)."""
        ingest_result = self.ingestion.run(source_path)
        processed = ingest_result.get("processed_elements", [])
        self._last_processed_elements = processed

        self.knowledge_graph = self.extraction_agent.build_graph(processed)
        self.vector_index.build(processed)

        return {
            "ingestion_errors": ingest_result.get("errors", []),
            "graph_stats": self.knowledge_graph.stats(),
            "num_vector_entries": len(self.vector_index),
        }

    def save(self, graph_path: str, elements_path: Optional[str] = None) -> None:
        if self.knowledge_graph is None:
            raise RuntimeError("Nothing ingested yet — call ingest() first.")
        self.knowledge_graph.save(graph_path)
        if elements_path:
            Path(elements_path).write_text(
                json.dumps(self._last_processed_elements, indent=2, default=str), encoding="utf-8"
            )

    def load(self, graph_path: str, elements_path: Optional[str] = None) -> None:
        self.knowledge_graph = KnowledgeGraph.load(graph_path)
        if elements_path:
            self._last_processed_elements = json.loads(Path(elements_path).read_text(encoding="utf-8"))
            self.vector_index.build(self._last_processed_elements)
        else:
            logger.warning("No elements_path given — vector ('vector'/'hybrid') retrieval will return nothing.")

    # -- Objective 2, part 2: routing / retrieval / verification loop -----

    def ask(self, question: str) -> dict:
        if self.knowledge_graph is None:
            raise RuntimeError("Nothing ingested/loaded yet — call ingest() or load() first.")
        qa = QAPipeline(self.config, self.knowledge_graph, self.vector_index, ollama_client=self.llm)
        return qa.ask(question)
