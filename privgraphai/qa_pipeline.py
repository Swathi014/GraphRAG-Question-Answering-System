"""Question-answering graph — Objective 2."""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from .config import PrivGraphAIConfig
from .graph_store import KnowledgeGraph
from .ollama_client import OllamaClient
from .pipeline_state import PrivGraphAIState
from .routing_agent import RoutingAgent
from .vector_index import SimpleVectorIndex
from .verification_agent import VerificationAgent

logger = logging.getLogger("privgraphai.qa_pipeline")

_ANSWER_PROMPT = """Answer the QUESTION using ONLY the EVIDENCE below. If the evidence is
insufficient, say plainly what's missing rather than guessing.

QUESTION: {question}

EVIDENCE:
{evidence}

ANSWER:
"""


class QAPipeline:
    def __init__(
        self,
        config: PrivGraphAIConfig,
        knowledge_graph: KnowledgeGraph,
        vector_index: SimpleVectorIndex,
        ollama_client: Optional[OllamaClient] = None,
        routing_agent: Optional[RoutingAgent] = None,
        verification_agent: Optional[VerificationAgent] = None,
    ):
        self.config = config
        self.kg = knowledge_graph
        self.vector_index = vector_index
        self.llm = ollama_client or OllamaClient(config.ollama_host, config.request_timeout_s)
        self.router = routing_agent or RoutingAgent(config, self.llm)
        self.verifier = verification_agent or VerificationAgent(config, self.llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PrivGraphAIState)
        graph.add_node("route_question", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("verify_evidence", self._verify_node)
        graph.add_node("generate_answer", self._answer_node)

        graph.set_entry_point("route_question")
        graph.add_edge("route_question", "retrieve")
        graph.add_edge("retrieve", "verify_evidence")
        graph.add_conditional_edges(
            "verify_evidence",
            self._should_retry,
            {"retry": "retrieve", "answer": "generate_answer"},
        )
        graph.add_edge("generate_answer", END)
        return graph.compile()

    def _route_node(self, state: PrivGraphAIState) -> dict:
        strategy = self.router.route(state["question"])
        return {"strategy": strategy, "retry_count": 0}

    def _retrieve_node(self, state: PrivGraphAIState) -> dict:
        retry_count = state.get("retry_count", 0)
        strategy = "hybrid" if retry_count > 0 else state.get("strategy", "hybrid")
        question = state["question"]

        retrieved: list[dict] = []
        if strategy in ("graph", "hybrid"):
            seed_ids = self.kg.find_entities(question)
            retrieved.extend(self.kg.multi_hop_elements(seed_ids, max_hops=self.config.graph_max_hops))
        if strategy in ("vector", "hybrid"):
            retrieved.extend(self.vector_index.search(question, top_k=3))

        deduped = self._dedupe(retrieved)
        return {"retrieved": deduped, "retry_count": retry_count + 1, "strategy": strategy}

    def _verify_node(self, state: PrivGraphAIState) -> dict:
        evidence_text = "\n\n".join(item.get("content", "") for item in state.get("retrieved", []))
        sufficient, missing = self.verifier.verify(state["question"], evidence_text)
        return {"sufficient": sufficient, "missing": missing}

    def _answer_node(self, state: PrivGraphAIState) -> dict:
        evidence_text = "\n\n".join(item.get("content", "") for item in state.get("retrieved", []))
        answer = self.llm.generate_text(
            _ANSWER_PROMPT.format(question=state["question"], evidence=evidence_text[:4000]),
            self.config.text_model,
            temperature=0.1,
        )
        return {"answer": answer}

    def _should_retry(self, state: PrivGraphAIState) -> str:
        if state.get("sufficient"):
            return "answer"
        if state.get("retry_count", 0) >= self.config.qa_max_retries:
            logger.info("Max retries reached without sufficient evidence; answering with best effort.")
            return "answer"
        return "retry"

    @staticmethod
    def _dedupe(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for item in items:
            eid = item.get("element_id")
            if eid and eid not in seen:
                seen.add(eid)
                out.append(item)
        return out

    def ask(self, question: str) -> PrivGraphAIState:
        initial: PrivGraphAIState = {"question": question, "errors": []}
        return self.graph.invoke(initial)
