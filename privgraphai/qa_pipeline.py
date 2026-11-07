"""Question-answering graph - Objective 2.

PROPOSED version. What changed and why (all found by reading the pilot logs and this file):
  * BUG: the answer step cut the evidence to its first 4000 characters, and the verifier to
    3000, after sorting it by kind and document number. In the survey pilot the evidence
    averaged 9,500 characters, so most retrieved passages never reached the model, and the
    ones that did were chosen by position, not relevance. Now the evidence is chosen by
    retrieval rank within a character budget, and only then put back in document order.
  * Retrieval order: vector hits (ranked by similarity) come before graph expansion, so the
    best passages are the ones kept when the budget is tight.
  * Answer prompt: the old one said "say plainly what's missing rather than guessing", which
    pushed the 3B model to refuse even when the answer was present. The new one asks for a
    short direct answer with element ids, and a fixed short sentence if truly absent.
  * Retry: the old retry repeated nearly the same search. Now the retry widens it
    (more vector hits and one more graph hop).
  * Round 2 (after the second pilot): the "graph" route no longer returns only the graph's
    few elements; a small vector backstop is added so the model is never left without evidence.
    The element-id citation instruction was removed from the answer prompt, because the 3B model
    sometimes answered with the id itself ("The offset is [text-3]"). The evidence budget is
    now 6000 characters by default and can be changed with the environment variable
    PRIVGRAPH_EVIDENCE_CHARS, so the quality/latency trade-off can be measured.
  * Round 3: the answer prompt is now "prompt B" from the experiments (complete sentences with every
    requested value, calculation steps first, "say briefly what is missing" instead of a fixed refusal).
    It scored 0.83 on the test document and 0.57 on the survey, against 0.67 for the terser prompt.
The graph structure, state keys and public methods are unchanged.
"""

from __future__ import annotations

import logging
import os
import re
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

_ID_RE = re.compile(r"^(.*?)-(\d+)(?:_(\d+))?$")   # text-12, text-12_3 (chunk), table-3_12 (row)
_VECTOR_TOP_K = 8
_RETRY_VECTOR_TOP_K = 16
_GRAPH_BACKSTOP_TOP_K = 4    # vector hits added after graph evidence on the "graph" route
# Evidence budget in characters (about 4 characters per token). Prompt-processing time grows with
# it: 9000 characters gave the best accuracy in the second pilot but a median latency of about 80 s.
_MAX_EVIDENCE_CHARS = int(os.environ.get("PRIVGRAPH_EVIDENCE_CHARS", "6000"))
_MIN_ITEM_CHARS = 25         # when the budget is tight, bare headings and fragments go first

_ANSWER_PROMPT = """Use the EVIDENCE to answer the QUESTION. Use only the evidence.
Write one or two complete sentences that include every value the question asks for. If a calculation is needed, write the steps first, then the result. Do not mention element ids or square brackets.
If the evidence does not contain the answer, say briefly what is missing.

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
        retrying = retry_count > 0
        strategy = "hybrid" if retrying else state.get("strategy", "hybrid")
        question = state["question"]
        top_k = _RETRY_VECTOR_TOP_K if retrying else _VECTOR_TOP_K
        hops = self.config.graph_max_hops + (1 if retrying else 0)

        retrieved: list[dict] = []
        if strategy == "vector":
            retrieved.extend(self.vector_index.search(question, top_k=top_k))
        elif strategy == "hybrid":
            # Vector hits first: they are ranked by similarity, so they are the best-first part.
            retrieved.extend(self.vector_index.search(question, top_k=top_k))
            seed_ids = self.kg.find_entities(question)
            retrieved.extend(self.kg.multi_hop_elements(seed_ids, max_hops=hops))
        else:  # "graph": graph evidence first, plus a small vector backstop
            seed_ids = self.kg.find_entities(question)
            retrieved.extend(self.kg.multi_hop_elements(seed_ids, max_hops=hops))
            retrieved.extend(self.vector_index.search(question, top_k=_GRAPH_BACKSTOP_TOP_K))

        deduped = self._dedupe(retrieved)
        return {"retrieved": deduped, "retry_count": retry_count + 1, "strategy": strategy}

    def _verify_node(self, state: PrivGraphAIState) -> dict:
        evidence_text = self._evidence_text(state.get("retrieved", []))
        sufficient, missing = self.verifier.verify(state["question"], evidence_text)
        return {"sufficient": sufficient, "missing": missing}

    def _answer_node(self, state: PrivGraphAIState) -> dict:
        evidence_text = self._evidence_text(state.get("retrieved", []))
        answer = self.llm.generate_text(
            _ANSWER_PROMPT.format(question=state["question"], evidence=evidence_text),
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
    def _evidence_text(items: list[dict], max_chars: int = _MAX_EVIDENCE_CHARS) -> str:
        """Evidence text labelled with element ids, in document order.

        `items` arrive best-first. If everything fits in `max_chars` it is all kept.
        Otherwise items are taken in rank order until the budget is used (bare headings and
        fragments skipped), and only the chosen ones are put back in document order.
        """

        def sort_key(item: dict):
            eid = str(item.get("element_id", ""))
            m = _ID_RE.match(eid)
            if m:
                return (m.group(1), int(m.group(2)), int(m.group(3) or 0))
            return (eid, 0, 0)

        def render(item: dict, limit: Optional[int] = None) -> str:
            content = item.get("content", "") or ""
            if limit is not None:
                content = content[:limit]
            return f"[{item.get('element_id')}] {content}"

        total = sum(len(render(i)) + 2 for i in items)
        if total <= max_chars:
            chosen = list(items)
        else:
            chosen, used = [], 0
            for item in items:
                if len((item.get("content") or "").strip()) < _MIN_ITEM_CHARS:
                    continue
                size = len(render(item)) + 2
                if size > max_chars - used:
                    if not chosen:  # a single huge element: keep its beginning
                        item = {**item, "content": (item.get("content") or "")[: max_chars - 40]}
                        chosen.append(item)
                        used = max_chars
                    continue
                chosen.append(item)
                used += size

        return "\n\n".join(render(item) for item in sorted(chosen, key=sort_key))

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
