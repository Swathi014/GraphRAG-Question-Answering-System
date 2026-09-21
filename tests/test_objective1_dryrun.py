"""
Dry-run test for Objective 1's LangGraph orchestration.

Docling and a running Ollama server aren't available in every environment
(CI, a reviewer's machine, this sandbox), so this test injects fake
implementations of both via PrivGraphAIPipeline's constructor. That proves
the *graph wiring* -- parse_document -> process_elements -> END, state
threaded through correctly -- is correct independent of the heavy runtime
dependencies. Swap in the real DoclingParser / OllamaClient to exercise the
full on-device path.
"""

from privgraphai.docling_parser import MultimodalElement
from privgraphai.pipeline import PrivGraphAIPipeline


class FakeDoclingParser:
    def parse(self, source):
        return [
            MultimodalElement("text-0", "text", 1, "PrivGraphAI runs entirely on-device."),
            MultimodalElement("table-0", "table", 2, "| A | B |\n|---|---|\n| 1 | 2 |"),
            MultimodalElement("figure-0", "figure", 3, "", image_path="fake.png"),
            MultimodalElement("formula-0", "formula", 4, "E = mc^2"),
        ]


class FakeOllamaClient:
    def is_available(self):
        return True

    def generate_text(self, prompt, model, **kwargs):
        return f"[normalized by {model}]"

    def describe_image(self, image_path, prompt, model):
        return f"[caption by {model} for {image_path}]"


def test_pipeline_runs_end_to_end_with_fakes():
    pipeline = PrivGraphAIPipeline(
        docling_parser=FakeDoclingParser(),
        ollama_client=FakeOllamaClient(),
    )
    result = pipeline.run("fake-document.pdf")

    assert result["errors"] == []
    assert len(result["processed_elements"]) == 4

    by_id = {e["element_id"]: e for e in result["processed_elements"]}
    assert "normalized by" in by_id["table-0"]["ai_description"]
    assert "normalized by" in by_id["formula-0"]["ai_description"]
    assert "caption by" in by_id["figure-0"]["ai_description"]
    assert by_id["text-0"]["ai_description"] is None  # plain text needs no local-model pass


def test_pipeline_records_parse_errors_without_crashing():
    class BrokenParser:
        def parse(self, source):
            raise RuntimeError("simulated Docling failure")

    pipeline = PrivGraphAIPipeline(docling_parser=BrokenParser(), ollama_client=FakeOllamaClient())
    result = pipeline.run("fake-document.pdf")

    assert result["processed_elements"] == []
    assert any("simulated Docling failure" in e for e in result["errors"])


if __name__ == "__main__":
    test_pipeline_runs_end_to_end_with_fakes()
    test_pipeline_records_parse_errors_without_crashing()
    print("OK: both dry-run tests passed.")
