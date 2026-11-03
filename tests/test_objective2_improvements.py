"""Tests for the verifier, evidence budgeting, LaTeX tidying and figure-text grouping.

Pure-logic tests: no Ollama, Docling or network needed.
"""

from types import SimpleNamespace as NS

from privgraphai.config import DEFAULT_CONFIG
from privgraphai.docling_parser import DoclingParser, _tidy_latex
from privgraphai.qa_pipeline import QAPipeline
from privgraphai.verification_agent import VerificationAgent


class ScriptedLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def generate_text(self, prompt, model, temperature=0.0):
        return self.outputs.pop(0)


EVIDENCE = "[text-3] The pilot was led by Dr. R. Menon.\n\n[table-0] | G2 | S-102 | -0.8 | 0.9 |"


def _verify(output: str) -> bool:
    agent = VerificationAgent(DEFAULT_CONFIG, ScriptedLLM(output))
    return agent.verify("Who led the pilot?", EVIDENCE)[0]


# ---------------------------------------------------------------- verifier
def test_verifier_accepts_a_quote_that_is_in_the_evidence():
    assert _verify('{"quotes": ["The pilot was led by Dr. R. Menon."]}') is True


def test_verifier_rejects_an_invented_quote():
    assert _verify('{"quotes": ["The pilot was funded by the ministry of agriculture in 2020."]}') is False


def test_verifier_rejects_a_quote_with_a_wrong_number():
    assert _verify('{"quotes": ["| G2 | S-102 | -0.8 | 1.9 |"]}') is False


def test_verifier_accepts_a_quote_with_minor_punctuation_changes():
    assert _verify('{"quotes": ["The pilot was led by Dr R Menon"]}') is True


def test_verifier_rejects_empty_quotes():
    assert _verify('{"quotes": []}') is False


def test_verifier_still_understands_the_old_json_format():
    assert _verify('{"sufficient": true, "missing": ""}') is True
    assert _verify('{"sufficient": "false", "missing": "x"}') is False


# ------------------------------------------------------ evidence budgeting
def _items(n, size=1000):
    # best-first order: text-n is the most relevant, text-1 the least
    return [{"element_id": f"text-{i}", "content": "x" * size} for i in range(n, 0, -1)]


def test_evidence_keeps_everything_when_it_fits():
    items = [{"element_id": "text-2", "content": "b"}, {"element_id": "text-1", "content": "a"}]
    assert QAPipeline._evidence_text(items) == "[text-1] a\n\n[text-2] b"


def test_evidence_keeps_best_ranked_items_within_budget_and_restores_document_order():
    text = QAPipeline._evidence_text(_items(30), max_chars=5000)
    ids = [block.split("]")[0][1:] for block in text.split("\n\n")]
    assert len(text) <= 5000 + 100
    assert "text-30" in ids                      # the most relevant item survives
    assert "text-1" not in ids                   # the least relevant one is dropped
    numbers = [int(i.split("-")[1]) for i in ids]
    assert numbers == sorted(numbers)            # shown in document order


def test_evidence_drops_bare_headings_first_when_over_budget():
    items = _items(8) + [{"element_id": "text-99", "content": "3 Heading"}]
    text = QAPipeline._evidence_text(items, max_chars=3500)
    assert "[text-99]" not in text


# ------------------------------------------------------------- parser bits
def test_tidy_latex_collapses_letter_spacing():
    assert _tidy_latex(r"H _ { c } = H _ { r a w } - b - d \, t") == r"H_{c} = H_{raw} - b - d \, t"


def _bbox(l, r, t, b):
    return NS(l=l, r=r, t=t, b=b, coord_origin="CoordOrigin.BOTTOMLEFT")


def _text(txt, label, page, box):
    return NS(text=txt, label=label, prov=[NS(page_no=page, bbox=box)])


def test_figure_text_is_grouped_by_column_and_removed_from_loose_text(tmp_path):
    texts = [
        _text("71.8", "text", 2, _bbox(260, 280, 465, 455)),
        _text("62.4", "text", 2, _bbox(190, 210, 435, 425)),
        _text("G1", "text", 2, _bbox(190, 210, 225, 215)),
        _text("G2", "text", 2, _bbox(260, 280, 225, 215)),
        _text("Figure 1: caption", "caption", 2, _bbox(100, 400, 150, 130)),
        _text("1", "page_footer", 2, _bbox(290, 300, 30, 20)),
    ]
    picture = NS(prov=[NS(page_no=2, bbox=_bbox(100, 400, 500, 190))])
    doc = NS(texts=texts, tables=[], pictures=[picture])

    parser = DoclingParser(image_export_dir=tmp_path)
    figure_texts, inside = parser._figure_text_map(doc)
    assert figure_texts[0] == "Values by label, left to right: G1 = 62.4; G2 = 71.8"   # value stays with its label
    assert inside == {0, 1, 2, 3}

    loose = parser._extract_text_and_formulas(doc, skip=inside)
    assert [e.element_id for e in loose] == ["text-4"]   # caption kept, footer and figure text gone


# ------------------------------------------------------- graph route backstop
class _StubKG:
    def find_entities(self, question):
        return ["e1"]

    def multi_hop_elements(self, seeds, max_hops=1):
        return [{"element_id": "text-1", "content": "graph neighbour element, long enough to keep"}]


class _StubIndex:
    def search(self, question, top_k=8):
        return [{"element_id": f"text-{10 + i}", "content": "vector hit number %d, long enough" % i} for i in range(top_k)]


class _StubRouter:
    def route(self, question):
        return "graph"


def test_graph_route_adds_a_small_vector_backstop_after_graph_evidence():
    qa = QAPipeline(DEFAULT_CONFIG, _StubKG(), _StubIndex(), ollama_client=ScriptedLLM(),
                    routing_agent=_StubRouter(), verification_agent=NS(verify=lambda q, e: (True, "")))
    out = qa._retrieve_node({"question": "anything", "strategy": "graph", "retry_count": 0})
    ids = [item["element_id"] for item in out["retrieved"]]
    assert ids[0] == "text-1"            # graph evidence first
    assert len(ids) == 1 + 4             # plus four vector hits so the model is never starved
