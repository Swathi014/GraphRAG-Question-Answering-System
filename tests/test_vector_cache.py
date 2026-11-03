"""Tests for the batched, cached vector index and the Ollama client additions (no Ollama needed)."""

import math

import pytest

from privgraphai import ollama_client as oc
from privgraphai.ollama_client import OllamaClient
from privgraphai.vector_index import SimpleVectorIndex


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _vec(text: str):
    # tiny deterministic "embedding": letter counts of a, b, c
    return [float(text.count("a")), float(text.count("b")), float(text.count("c")) + 0.1]


class FakeLLM:
    def __init__(self, fail_batch=False):
        self.single_calls, self.batch_calls, self.fail_batch = 0, 0, fail_batch

    def embed(self, text, model):
        self.single_calls += 1
        return _vec(text)

    def embed_many(self, texts, model):
        self.batch_calls += 1
        if self.fail_batch:
            raise RuntimeError("batch endpoint down")
        return [_vec(t) for t in texts]


def _elements(n=40):
    return [{"element_id": f"text-{i}", "element_type": "text", "content": "a" * (i % 5 + 1) + "b" * (i % 3)} for i in range(n)]


# ------------------------------------------------------------------ client
def test_embed_many_sends_a_list_and_returns_vectors_in_order(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["url"], seen["json"] = url, json
        return FakeResponse({"embeddings": [[1.0], [2.0]]})

    monkeypatch.setattr(oc.requests, "post", fake_post)
    out = OllamaClient("http://x:1").embed_many(["one", "two"], "nomic-embed-text")
    assert out == [[1.0], [2.0]]
    assert seen["url"].endswith("/api/embed") and seen["json"]["input"] == ["one", "two"]
    assert seen["json"]["keep_alive"] == "30m"


def test_embed_many_rejects_a_count_mismatch(monkeypatch):
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: FakeResponse({"embeddings": [[1.0]]}))
    with pytest.raises(ValueError):
        OllamaClient("http://x:1").embed_many(["one", "two"], "m")


def test_generate_text_passes_a_token_cap_only_when_asked(monkeypatch):
    payloads = []
    monkeypatch.setattr(oc.requests, "post", lambda url, json=None, timeout=None: payloads.append(json) or FakeResponse({"response": " ok "}))
    client = OllamaClient("http://x:1")
    assert client.generate_text("p", "m") == "ok"
    client.generate_text("p", "m", num_predict=700)
    assert "num_predict" not in payloads[0]["options"]
    assert payloads[1]["options"]["num_predict"] == 700


# ------------------------------------------------------------- vector index
def test_build_uses_batches_not_one_call_per_element():
    llm = FakeLLM()
    index = SimpleVectorIndex(llm, "m")
    index.build(_elements(40))
    assert len(index) == 40
    assert llm.batch_calls == 3 and llm.single_calls == 0      # 40 elements in batches of 16


def test_build_falls_back_to_single_calls_if_the_batch_call_fails():
    llm = FakeLLM(fail_batch=True)
    index = SimpleVectorIndex(llm, "m")
    index.build(_elements(20))
    assert len(index) == 20 and llm.single_calls == 20


def test_saved_index_is_reused_without_any_embedding_calls(tmp_path):
    cache = tmp_path / "elements.vectors.json"
    first = FakeLLM()
    SimpleVectorIndex(first, "m").build_or_load(_elements(30), cache)
    assert cache.exists() and first.batch_calls == 2

    second = FakeLLM()
    index = SimpleVectorIndex(second, "m")
    index.build_or_load(_elements(30), cache)
    assert len(index) == 30
    assert second.batch_calls == 0 and second.single_calls == 0   # loaded from disk


def test_saved_index_is_rebuilt_when_the_elements_or_the_model_change(tmp_path):
    cache = tmp_path / "v.json"
    SimpleVectorIndex(FakeLLM(), "m").build_or_load(_elements(10), cache)

    changed = _elements(10)
    changed[3]["content"] = "completely different text"
    llm = FakeLLM()
    SimpleVectorIndex(llm, "m").build_or_load(changed, cache)
    assert llm.batch_calls == 1                                    # rebuilt

    llm2 = FakeLLM()
    SimpleVectorIndex(llm2, "other-model").build_or_load(changed, cache)
    assert llm2.batch_calls == 1                                   # different model -> rebuilt


def test_search_returns_the_most_similar_element_first(tmp_path):
    llm = FakeLLM()
    index = SimpleVectorIndex(llm, "m")
    index.build([{"element_id": "e-a", "content": "aaaa"}, {"element_id": "e-b", "content": "bbbb"}])
    hits = index.search("aaa", top_k=2)
    assert [h["element_id"] for h in hits] == ["e-a", "e-b"]
    assert hits[0]["score"] > hits[1]["score"] and math.isclose(hits[0]["score"], hits[0]["score"])
