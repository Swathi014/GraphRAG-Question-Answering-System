"""Tests for paragraph chunking, per-row tables, row handling in the pipeline, and evidence ordering."""

from types import SimpleNamespace as NS

import pandas as pd

from privgraphai.config import DEFAULT_CONFIG
from privgraphai.docling_parser import DoclingParser, _split_text
from privgraphai.pipeline import PrivGraphAIPipeline
from privgraphai.qa_pipeline import QAPipeline


def _paragraph(n_sentences=30):
    return " ".join(f"Sentence number {i} says something specific about topic {i}." for i in range(n_sentences))


# ------------------------------------------------------------- text chunks
def test_short_text_is_not_split():
    assert _split_text("A short sentence.") == ["A short sentence."]


def test_long_text_is_split_with_overlap_and_nothing_is_lost():
    text = _paragraph(30)
    chunks = _split_text(text, max_chars=400, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 400 + 80 for c in chunks)
    for i in range(30):                                   # every sentence is in at least one chunk
        assert any(f"Sentence number {i} says" in c for c in chunks)
    first_sentences = chunks[1].split(". ")[0]
    assert first_sentences in chunks[0]                  # consecutive chunks overlap


def test_a_single_huge_run_without_sentence_ends_is_hard_split():
    chunks = _split_text("x" * 2500, max_chars=900, overlap=100)
    assert len(chunks) == 3 and all(len(c) <= 900 for c in chunks)


def test_parser_gives_long_paragraphs_chunk_ids_and_leaves_short_ones_alone(tmp_path):
    def item(text, label="text"):
        return NS(text=text, label=label, prov=[NS(page_no=3, bbox=NS(l=0, r=1, t=1, b=0, coord_origin="X"))])
    doc = NS(texts=[item(_paragraph(30)), item("3 Heading", "section_header")], tables=[], pictures=[])
    parser = DoclingParser(image_export_dir=tmp_path)
    els = parser._extract_text_and_formulas(doc)
    ids = [e.element_id for e in els]
    assert ids[0] == "text-0_1" and ids[-1] == "text-1" and "text-0_2" in ids
    assert all(e.page_number == 3 for e in els)
    assert els[0].metadata["parent"] == "text-0"


# ------------------------------------------------------------------ tables
class FakeTable:
    def __init__(self, df, markdown, caption="Table II: Retrieval techniques"):
        self._df, self._md, self._caption = df, markdown, caption
        self.prov = [NS(page_no=13)]

    def export_to_markdown(self, doc=None):
        return self._md

    def export_to_dataframe(self, doc=None):
        return self._df

    def caption_text(self, doc):
        return self._caption


def test_large_table_becomes_one_element_per_row(tmp_path):
    df = pd.DataFrame(
        [["G-Retriever", "k-nearest neighbors", "PCST", "subgraph"], ["RoG", "breadth-first search", "LLM agent", "reasoning path"]] * 20,
        columns=["Model", "Matching", "Pruning", "Output"],
    )
    parser = DoclingParser(image_export_dir=tmp_path)
    els = parser._extract_tables(NS(tables=[FakeTable(df, "x" * 3000)]))
    assert els[0].element_id == "table-0" and els[0].metadata["table_overview"]
    rows = [e for e in els if e.metadata.get("table_row")]
    assert len(rows) == 40 and rows[0].element_id == "table-0_1"
    assert "G-Retriever" in rows[0].content and "Pruning = PCST" in rows[0].content
    assert "Table II" in rows[0].content


def test_small_table_stays_one_element(tmp_path):
    df = pd.DataFrame([["a", "b"]], columns=["x", "y"])
    parser = DoclingParser(image_export_dir=tmp_path)
    els = parser._extract_tables(NS(tables=[FakeTable(df, "| x | y |\n| a | b |")]))
    assert [e.element_id for e in els] == ["table-0"] and not els[0].metadata


# ---------------------------------------------------------------- pipeline
def test_row_and_overview_elements_do_not_trigger_an_llm_call():
    pipe = PrivGraphAIPipeline.__new__(PrivGraphAIPipeline)
    pipe.config = DEFAULT_CONFIG
    pipe.llm = __import__("privgraphai.ollama_client", fromlist=["OllamaClient"]).OllamaClient()  # raises if generate_text is called
    from privgraphai.docling_parser import MultimodalElement
    row = MultimodalElement("table-0_1", "table", 13, "Table II - row 1: Model = RoG", metadata={"table_row": True})
    overview = MultimodalElement("table-0", "table", 13, "Table II. Columns: Model", metadata={"table_overview": True})
    assert pipe._process_single_element(row)["ai_description"] is None
    assert pipe._process_single_element(overview)["ai_description"] is None


# ------------------------------------------------------- evidence ordering
def test_evidence_keeps_document_order_for_chunk_and_row_ids():
    items = [
        {"element_id": "text-10_2", "content": "c"}, {"element_id": "table-3_12", "content": "r12"},
        {"element_id": "text-9", "content": "n"}, {"element_id": "text-10_1", "content": "b"},
        {"element_id": "table-3_2", "content": "r2"}, {"element_id": "text-10", "content": "whole"},
    ]
    ids = [b.split("]")[0][1:] for b in QAPipeline._evidence_text(items).split("\n\n")]
    assert ids == ["table-3_2", "table-3_12", "text-9", "text-10", "text-10_1", "text-10_2"]
