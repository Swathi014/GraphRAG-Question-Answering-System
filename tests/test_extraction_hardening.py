"""Tests for the hardened extraction agent and graph store (no Ollama needed)."""

from privgraphai.config import DEFAULT_CONFIG
from privgraphai.extraction_agent import ExtractionAgent, _is_reference_like
from privgraphai.graph_store import KnowledgeGraph


class ScriptedLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate_text(self, prompt, model, temperature=0.0):
        self.calls += 1
        return self.outputs.pop(0)


def _els(*texts):
    return [{"element_id": f"text-{i}", "element_type": "text", "content": t, "page_number": 1}
            for i, t in enumerate(texts)]


def test_list_valued_relation_fields_do_not_crash_ingestion():
    bad = ('{"entities": [{"name": "Alpha", "type": "method"}, {"name": ["x", "y"], "type": "m"}, "just a string"],'
           ' "relations": [{"source": "Alpha", "relation": "uses", "target": ["Beta", "Gamma"]},'
           ' {"source": "Alpha", "relation": "beats", "target": "Beta"}, 5]}')
    kg = ExtractionAgent(DEFAULT_CONFIG, ScriptedLLM(bad)).build_graph(_els("Alpha uses Beta and Gamma in the study."))
    names = {d["name"] for _, d in kg.graph.nodes(data=True) if d.get("kind") == "entity"}
    assert names == {"Alpha", "Beta"}            # the good relation survived; the bad ones were dropped
    labels = [d["relation"] for _, _, d in kg.graph.edges(data=True) if d["relation"] != "MENTIONED_IN"]
    assert labels == ["beats"]


def test_wrong_top_level_types_are_ignored():
    kg = ExtractionAgent(DEFAULT_CONFIG, ScriptedLLM('{"entities": "none", "relations": {"a": 1}}')).build_graph(
        _els("Some ordinary sentence long enough to be extracted."))
    assert kg.stats()["num_entities"] == 0


def test_reference_entries_are_not_sent_to_the_model():
    ref = ("[12] Aaron Smith, Beth Jones, Carl Wu, Dana Lee, Eli Park, Fay Chen, Gus Roy. 2024. A long title. "
           "In Proceedings of the Conference on Things. arXiv:2401.00001")
    llm = ScriptedLLM('{"entities": [], "relations": []}')
    kg = ExtractionAgent(DEFAULT_CONFIG, llm).build_graph(_els(ref, "A normal sentence long enough to extract from."))
    assert llm.calls == 1                         # only the normal sentence reached the model
    assert kg.graph.has_node("text-0")            # the reference element is still in the graph
    assert _is_reference_like(ref) and not _is_reference_like("A normal sentence long enough to extract from.")


def test_graph_store_rejects_non_string_names_without_leaving_orphans():
    kg = KnowledgeGraph()
    try:
        kg.add_relation("Alpha", "uses", ["Beta"])
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert kg.stats()["num_nodes"] == 0


def test_find_entities_uses_whole_words_and_ignores_tiny_names():
    kg = KnowledgeGraph()
    for name in ("RAG", "ti", "retrieval", "Self-RAG", "grapher"):
        kg.add_entity(name)
    found = {kg.graph.nodes[n]["name"] for n in kg.find_entities("How does Self-RAG improve retrieval in practice?")}
    # "ti" is too short; "grapher" is not a whole word in the question; "RAG" is a whole word inside "Self-RAG"
    assert found == {"RAG", "Self-RAG", "retrieval"}


def test_multi_hop_elements_are_ranked_by_distance_and_shared_entities():
    kg = KnowledgeGraph()
    for i in (1, 2, 3):
        kg.add_element(f"text-{i}", "text", f"element {i} content that is long enough")
    kg.link_entity_to_element("Alpha", "text-1")
    kg.link_entity_to_element("Beta", "text-1")       # text-1 mentions both seeds
    kg.link_entity_to_element("Alpha", "text-2")      # text-2 mentions one seed
    kg.add_relation("Alpha", "uses", "Gamma")         # traversal follows extracted relations
    kg.link_entity_to_element("Gamma", "text-3")      # so text-3 is two hops from Alpha
    seeds = [kg.add_entity("Alpha"), kg.add_entity("Beta")]
    ids = [e["element_id"] for e in kg.multi_hop_elements(seeds, max_hops=3)]
    assert ids[:2] == ["text-1", "text-2"]            # directly linked first, shared entity first
    assert ids.index("text-3") > ids.index("text-2")  # the two-hop element comes after
