# PrivGraphAI — Objectives 1 & 2

> **Objective 1:** Design and implement PrivGraphAI, integrating Docling,
> quantized models served via Ollama, and LangGraph agent orchestration for
> on-device multimodal processing.
>
> **Objective 2:** Construct a multimodal knowledge graph via specialized
> extraction agents, with routing and evidence-verification agents enabling
> reliable multi-hop reasoning.

This package is self-contained (Objective 1 is carried over unchanged from
its own delivery) so it can be run independently of anything else.

## What's inside

```
privgraphai/
  config.py              # model names, Ollama host, graph/retry settings
  docling_parser.py       # Docling wrapper -> MultimodalElement list        (Obj 1)
  ollama_client.py        # local Ollama REST client: text, vision, embed    (Obj 1+2)
  pipeline_state.py        # shared LangGraph state schema
  pipeline.py               # ingestion graph: parse_document -> process_elements  (Obj 1)
  graph_store.py            # in-memory multimodal knowledge graph (NetworkX)       (Obj 2)
  extraction_agent.py       # builds the graph from Obj 1's processed elements     (Obj 2)
  vector_index.py           # local embedding-based retrieval path                  (Obj 2)
  routing_agent.py          # graph / vector / hybrid strategy selection            (Obj 2)
  verification_agent.py     # judges evidence sufficiency, drives the retry loop   (Obj 2)
  qa_pipeline.py             # QA graph: route -> retrieve -> verify -> (retry|answer) (Obj 2)
  system.py                  # facade: ingest() / save() / load() / ask()
  cli.py                      # `python -m privgraphai.cli ingest|ask ...`
tests/
  test_objective1_dryrun.py   # ingestion graph wiring, with fakes
  test_objective2_dryrun.py   # extraction, routing, verification, and the QA retry loop
requirements.txt
```

## How Objective 2 actually works

**Knowledge graph construction** (`extraction_agent.py` + `graph_store.py`):
for every element Objective 1 produced, a local LLM call extracts named
entities and relations as JSON, which get written into a `KnowledgeGraph`
(a NetworkX `MultiDiGraph`). Entities are nodes; elements are nodes;
`MENTIONED_IN` edges connect an entity to the elements it appears in;
labeled edges connect entities to each other.

**Routing agent** (`routing_agent.py`): given a question, asks the LLM to
classify it as needing `"graph"` (multi-hop, comparisons), `"vector"`
(single-fact lookup), or `"hybrid"` retrieval — falling back to a keyword
heuristic if the model's JSON doesn't parse, so this step never blocks.

**Retrieval** (`qa_pipeline.py::_retrieve_node`): for `"graph"`, does a
breadth-first walk outward from entities matching the question, up to
`config.graph_max_hops`, collecting every element reached. For `"vector"`,
does embedding-similarity search over a `SimpleVectorIndex` (Ollama's
`nomic-embed-text`, cosine similarity). `"hybrid"` does both and merges.

**Evidence-verification agent** (`verification_agent.py`): the LLM judges
whether the retrieved evidence is actually sufficient to answer the
question. If not, the graph loops back to `retrieve` — this time forced to
`"hybrid"` — for up to `config.qa_max_retries` attempts, then answers with
whatever evidence it has rather than looping forever.

This loop is the literal implementation of the project abstract's line:
*"an evidence-verification agent checks the relevance and adequacy of
retrieved information and initiates further retrieval when necessary."*

## Setup

1. **Install Ollama**, start it, and pull the three local models:
   ```bash
   ollama serve
   ollama pull qwen2.5:3b-instruct-q4_K_M   # text: extraction/routing/verification/answers
   ollama pull moondream                     # vision: figure captioning (Objective 1)
   ollama pull nomic-embed-text              # embeddings (Objective 2's vector path)
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# 1. Ingest a document: extraction (Obj 1) + knowledge graph + vector index (Obj 2)
python -m privgraphai.cli ingest path/to/document.pdf --graph graph.json --elements elements.json

# 2. Ask questions against the graph you just built
python -m privgraphai.cli ask "How does the routing agent decide between graph and vector retrieval?" \
    --graph graph.json --elements elements.json
```

Or from Python directly:

```python
from privgraphai import PrivGraphAISystem

system = PrivGraphAISystem()
system.ingest("path/to/document.pdf")
result = system.ask("What models does PrivGraphAI use locally?")
print(result["answer"])
```

## Running the tests

Both objectives' dry-run tests use fake/scripted Ollama clients so they run
without Docling installed or an Ollama server up:

```bash
pip install pytest
pytest tests/ -v      # 9 tests: 2 for Objective 1, 7 for Objective 2
```

The Objective 2 tests are worth reading closely if you're extending this —
`test_qa_pipeline_retries_once_then_answers` in particular scripts an
insufficient-then-sufficient verification sequence to prove the LangGraph
conditional loop genuinely re-enters `retrieve`, not just `verify`, and
`test_qa_pipeline_stops_retrying_at_max_retries_and_still_answers` proves
the loop terminates and still produces an answer even when evidence never
becomes sufficient.

## Design notes

- **Every agent degrades gracefully.** Routing falls back to a keyword
  heuristic; verification treats a parse failure as "insufficient" rather
  than crashing; extraction skips an element it can't parse JSON for rather
  than aborting the whole graph build. Small quantized models on CPU don't
  always return clean JSON — the pipeline assumes that going in.
- **The retry loop always terminates.** `_should_retry` is a pure function
  of `state["sufficient"]` and `state["retry_count"]` against
  `config.qa_max_retries` — there's no path that loops indefinitely.
- **Conditional edges don't mutate state.** `_should_retry` only reads state
  and returns a routing key; the retry's strategy-widening (to `"hybrid"`)
  happens inside `_retrieve_node` itself, based on `retry_count`. Mutating
  state from inside a conditional-edge function is a common LangGraph
  mistake this code deliberately avoids.
- **`ingest` and `ask` are separable.** The knowledge graph and processed
  elements are saved to JSON, so asking a question doesn't require
  re-running Docling extraction — matches how you'd actually use this (parse
  once, query many times).
