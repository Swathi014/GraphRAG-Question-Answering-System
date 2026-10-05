"""
Dry-run and regression tests for Objective 2.

Docling and a running Ollama server aren't required — a FakeOllamaClient /
ScriptedOllamaClient stands in, driven by scripted responses so we can force
specific paths through the code: a JSON entity/relation extraction, a
routing decision, an insufficient-then-sufficient verification sequence (to
prove the retry loop actually loops), and a final answer.

Several tests below (the `_parse_json_relaxed` ones in particular) are built
directly from real failures captured while running Objective 2 against an
actual 20+ page research paper — not synthetic cases.
"""

import json

from privgraphai.extraction_agent import ExtractionAgent
from privgraphai.graph_store import KnowledgeGraph
from privgraphai.qa_pipeline import QAPipeline
from privgraphai.routing_agent import RoutingAgent
from privgraphai.vector_index import SimpleVectorIndex
from privgraphai.verification_agent import VerificationAgent


class ScriptedOllamaClient:
    """Returns pre-scripted generate_text() replies in order, per-model, and
    fixed embeddings, so tests can force specific code paths deterministically."""

    def __init__(self, scripted_replies=None, embeddings=None):
        self._replies = list(scripted_replies or [])
        self._embeddings = embeddings or {}

    def is_available(self):
        return True

    def generate_text(self, prompt, model, **kwargs):
        if not self._replies:
            raise AssertionError("ScriptedOllamaClient ran out of scripted replies")
        return self._replies.pop(0)

    def describe_image(self, image_path, prompt, model):
        return "[fake caption]"

    def embed(self, text, model):
        return self._embeddings.get(text, [0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# ExtractionAgent._parse_json_relaxed -- real failure cases seen in the wild
# --------------------------------------------------------------------------

def test_parse_json_relaxed_handles_clean_json():
    result = ExtractionAgent._parse_json_relaxed('{"entities": [{"name": "X", "type": "y"}], "relations": []}')
    assert result == {"entities": [{"name": "X", "type": "y"}], "relations": []}


def test_parse_json_relaxed_picks_the_real_object_over_an_empty_false_start():
    # Observed verbatim: the model emitted "{}" then, on the next line, the actual answer.
    raw = '{}\n{"entities": [], "relations": []}'
    result = ExtractionAgent._parse_json_relaxed(raw)
    assert result == {"entities": [], "relations": []}


def test_parse_json_relaxed_repairs_unclosed_relations_array():
    # Observed verbatim (NeoChip case): "relations" array opened with `[` but
    # never closed with `]` before the trailing `}}}`.
    raw = ('{"entities": [{"name": "NeoChip", "type": "organization"}], '
           '"relations": [{"source": "NeoChip", "relation": "shares", "target": "surged"}}}')
    result = ExtractionAgent._parse_json_relaxed(raw)
    assert result["entities"] == [{"name": "NeoChip", "type": "organization"}]
    assert result["relations"] == [{"source": "NeoChip", "relation": "shares", "target": "surged"}]


def test_parse_json_relaxed_repairs_unclosed_relations_array_second_example():
    # Observed verbatim (Microsoft case) -- same failure shape, different content.
    raw = ('{"entities": [{"name": "Microsoft", "type": "organization"}], '
           '"relations": [{"source": "Microsoft (2023)", "relation": "relatedTo", '
           '"target": "large language models"}}}')
    result = ExtractionAgent._parse_json_relaxed(raw)
    assert result["entities"] == [{"name": "Microsoft", "type": "organization"}]
    assert result["relations"][0]["target"] == "large language models"


def test_parse_json_relaxed_gives_up_gracefully_on_true_garbage():
    result = ExtractionAgent._parse_json_relaxed("not json at all, no braces here")
    assert result == {"entities": [], "relations": []}


def test_parse_json_relaxed_handles_empty_string():
    assert ExtractionAgent._parse_json_relaxed("") == {"entities": [], "relations": []}


# --------------------------------------------------------------------------
# ExtractionAgent -> KnowledgeGraph
# --------------------------------------------------------------------------

def test_extraction_agent_skips_trivial_short_elements():
    """Elements shorter than config.min_extraction_chars should never reach
    the LLM at all -- proven here by giving a client with zero scripted
    replies and confirming it's never called (would raise if it were)."""
    processed_elements = [
        {"element_id": "text-0", "element_type": "text", "content": "3", "ai_description": None},
        {"element_id": "text-1", "element_type": "text", "content": "Fig. 2", "ai_description": None},
    ]
    llm = ScriptedOllamaClient(scripted_replies=[])  # would raise AssertionError if called
    agent = ExtractionAgent(config=_FakeConfig(), ollama_client=llm)
    kg = agent.build_graph(processed_elements)
    assert kg.stats()["num_entities"] == 0


def test_extraction_agent_builds_graph_from_processed_elements():
    processed_elements = [
        {"element_id": "text-0", "element_type": "text", "content": "Qwen2.5 runs on Ollama.",
         "ai_description": None, "page_number": 1},
        {"element_id": "table-0", "element_type": "table", "content": "| model | ram |",
         "ai_description": "A table comparing model RAM usage.", "page_number": 2},
    ]

    extraction_reply = json.dumps({
        "entities": [{"name": "Qwen2.5", "type": "model"}, {"name": "Ollama", "type": "tool"}],
        "relations": [{"source": "Qwen2.5", "relation": "RUNS_ON", "target": "Ollama"}],
    })
    table_reply = json.dumps({"entities": [{"name": "RAM usage", "type": "metric"}], "relations": []})

    llm = ScriptedOllamaClient(scripted_replies=[extraction_reply, table_reply])
    agent = ExtractionAgent(config=_FakeConfig(), ollama_client=llm)

    kg = agent.build_graph(processed_elements)
    stats = kg.stats()

    assert stats["num_elements"] == 2
    assert stats["num_entities"] == 3  # Qwen2.5, Ollama, RAM usage
    seeds = kg.find_entities("qwen2.5 runs on ollama")
    assert seeds, "expected to find the Qwen2.5 entity"
    hop = kg.multi_hop_elements(seeds, max_hops=2)
    element_ids = {e["element_id"] for e in hop}
    assert "text-0" in element_ids  # direct mention
    assert "table-0" not in element_ids  # not reachable from Qwen2.5/Ollama


# --------------------------------------------------------------------------
# RoutingAgent
# --------------------------------------------------------------------------

def test_routing_agent_parses_llm_json():
    llm = ScriptedOllamaClient(scripted_replies=[json.dumps({"strategy": "graph", "reason": "multi-hop"})])
    agent = RoutingAgent(_FakeConfig(), llm)
    assert agent.route("How does X relate to Y?") == "graph"


def test_routing_agent_falls_back_to_heuristic_on_bad_json():
    llm = ScriptedOllamaClient(scripted_replies=["not json at all"])
    agent = RoutingAgent(_FakeConfig(), llm)
    # no multi-hop marker words -> heuristic should say "vector"
    assert agent.route("What is the RAM usage of Qwen2.5?") == "vector"


# --------------------------------------------------------------------------
# VerificationAgent
# --------------------------------------------------------------------------

def test_verification_agent_reports_insufficient_with_no_evidence():
    agent = VerificationAgent(_FakeConfig(), ScriptedOllamaClient())
    sufficient, missing = agent.verify("some question", "")
    assert sufficient is False
    assert missing


def test_verification_agent_coerces_string_booleans_correctly():
    """A small model may return "sufficient": "false" (a JSON string) instead
    of the JSON boolean false. bool("false") is True in Python, so a naive
    cast would silently flip this — confirm the agent handles both forms."""
    llm_string_false = ScriptedOllamaClient(scripted_replies=[json.dumps({"sufficient": "false", "missing": "x"})])
    sufficient, _ = VerificationAgent(_FakeConfig(), llm_string_false).verify("q", "some evidence")
    assert sufficient is False

    llm_string_true = ScriptedOllamaClient(scripted_replies=[json.dumps({"sufficient": "true", "missing": ""})])
    sufficient, _ = VerificationAgent(_FakeConfig(), llm_string_true).verify("q", "some evidence")
    assert sufficient is True

    llm_real_bool = ScriptedOllamaClient(scripted_replies=[json.dumps({"sufficient": True, "missing": ""})])
    sufficient, _ = VerificationAgent(_FakeConfig(), llm_real_bool).verify("q", "some evidence")
    assert sufficient is True


# --------------------------------------------------------------------------
# QAPipeline — the full LangGraph retry loop
# --------------------------------------------------------------------------

_ELEMENT_TEXT = "PrivGraphAI runs fully on-device."


def _build_populated_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_element("el-1", "text", _ELEMENT_TEXT)
    kg.link_entity_to_element("PrivGraphAI", "el-1", "system")
    return kg


def _build_populated_vector_index() -> SimpleVectorIndex:
    """Vector index with one matching entry, so a vector-only retrieval pass
    finds real (non-empty) evidence rather than legitimately finding nothing
    (which is what an *empty* index correctly returns)."""
    embed_llm = ScriptedOllamaClient(embeddings={_ELEMENT_TEXT: [1.0, 0.0], "What is PrivGraphAI?": [1.0, 0.0]})
    vindex = SimpleVectorIndex(embed_llm, "fake-embed-model")
    vindex.build([{"element_id": "el-1", "element_type": "text", "content": _ELEMENT_TEXT, "ai_description": None}])
    return vindex


def test_qa_pipeline_answers_directly_when_first_pass_is_sufficient():
    """route -> retrieve -> verify(sufficient) -> answer, no retry."""
    kg = _build_populated_graph()
    vindex = _build_populated_vector_index()

    llm = ScriptedOllamaClient(scripted_replies=[
        json.dumps({"strategy": "graph"}),                          # route_question
        json.dumps({"sufficient": True, "missing": ""}),            # verify_evidence
        "PrivGraphAI runs on-device.",                               # generate_answer
    ])

    qa = QAPipeline(_FakeConfig(), kg, vindex, ollama_client=llm)
    result = qa.ask("What is PrivGraphAI?")

    assert result["sufficient"] is True
    assert result["retry_count"] == 1  # retrieve ran exactly once
    assert result["answer"] == "PrivGraphAI runs on-device."
    assert len(result["retrieved"]) == 1


def test_qa_pipeline_retries_once_then_answers():
    """route -> retrieve -> verify(insufficient) -> retrieve(again, widened to
    hybrid) -> verify(sufficient) -> answer. Proves the LangGraph conditional
    loop actually re-enters `retrieve`, not just re-runs verification."""
    kg = _build_populated_graph()
    vindex = _build_populated_vector_index()

    llm = ScriptedOllamaClient(scripted_replies=[
        json.dumps({"strategy": "vector"}),                          # route_question (deliberately narrow)
        json.dumps({"sufficient": False, "missing": "need more"}),   # verify_evidence, pass 1
        json.dumps({"sufficient": True, "missing": ""}),             # verify_evidence, pass 2
        "Final answer after retry.",                                 # generate_answer
    ])

    qa = QAPipeline(_FakeConfig(), kg, vindex, ollama_client=llm)
    result = qa.ask("What is PrivGraphAI?")

    assert result["retry_count"] == 2       # retrieve ran twice
    assert result["strategy"] == "hybrid"   # widened on the retry
    assert result["sufficient"] is True
    assert result["answer"] == "Final answer after retry."


def test_qa_pipeline_stops_retrying_at_max_retries_and_still_answers():
    """Verification never reports sufficient -> loop must stop at
    config.qa_max_retries and still produce an answer, not loop forever."""
    kg = _build_populated_graph()
    vindex = _build_populated_vector_index()

    config = _FakeConfig(qa_max_retries=2)
    insufficient = json.dumps({"sufficient": False, "missing": "still missing"})
    llm = ScriptedOllamaClient(scripted_replies=[
        json.dumps({"strategy": "vector"}),   # route_question
        insufficient,                          # verify pass 1 -> retry (retry_count=1 < 2)
        insufficient,                          # verify pass 2 -> retry (retry_count=2 >= 2 -> stop)
        "Best-effort answer.",                 # generate_answer
    ])

    qa = QAPipeline(config, kg, vindex, ollama_client=llm)
    result = qa.ask("What is PrivGraphAI?")

    assert result["retry_count"] == 2
    assert result["sufficient"] is False
    assert result["answer"] == "Best-effort answer."


class _FakeConfig:
    """Minimal stand-in for PrivGraphAIConfig — avoids depending on Ollama
    model names that don't matter for these tests."""

    def __init__(self, qa_max_retries=2, graph_max_hops=2, min_extraction_chars=15):
        self.text_model = "fake-text-model"
        self.vision_model = "fake-vision-model"
        self.embedding_model = "fake-embed-model"
        self.qa_max_retries = qa_max_retries
        self.graph_max_hops = graph_max_hops
        self.min_extraction_chars = min_extraction_chars


if __name__ == "__main__":
    test_parse_json_relaxed_handles_clean_json()
    test_parse_json_relaxed_picks_the_real_object_over_an_empty_false_start()
    test_parse_json_relaxed_repairs_unclosed_relations_array()
    test_parse_json_relaxed_repairs_unclosed_relations_array_second_example()
    test_parse_json_relaxed_gives_up_gracefully_on_true_garbage()
    test_parse_json_relaxed_handles_empty_string()
    test_extraction_agent_skips_trivial_short_elements()
    test_extraction_agent_builds_graph_from_processed_elements()
    test_routing_agent_parses_llm_json()
    test_routing_agent_falls_back_to_heuristic_on_bad_json()
    test_verification_agent_reports_insufficient_with_no_evidence()
    test_verification_agent_coerces_string_booleans_correctly()
    test_qa_pipeline_answers_directly_when_first_pass_is_sufficient()
    test_qa_pipeline_retries_once_then_answers()
    test_qa_pipeline_stops_retrying_at_max_retries_and_still_answers()
    print("OK: all Objective 2 dry-run tests passed.")
