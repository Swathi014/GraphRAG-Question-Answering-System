# PrivGraphAI — Complete Project Code (Objectives 1 & 2 + Web UI)

A privacy-preserving, fully on-device Agentic GraphRAG framework for
multimodal document question answering. Everything in this package runs
locally — document parsing, model inference, and the control-panel UI all
stay on your machine with no cloud calls.

> **Objective 1:** Design and implement PrivGraphAI, integrating Docling,
> quantized models served via Ollama, and LangGraph agent orchestration for
> on-device multimodal processing.
>
> **Objective 2:** Construct a multimodal knowledge graph via specialized
> extraction agents, with routing and evidence-verification agents enabling
> reliable multi-hop reasoning.

## Project layout

```
privgraphai/               <- the core package (Objectives 1 & 2)
  config.py                   model names, Ollama host, graph/retry settings
  docling_parser.py           Docling wrapper -> MultimodalElement list        (Obj 1)
  ollama_client.py            local Ollama REST client: text, vision, embed    (Obj 1+2)
  pipeline_state.py           shared LangGraph state schema
  pipeline.py                 ingestion graph: parse_document -> process_elements  (Obj 1)
  graph_store.py              in-memory multimodal knowledge graph (NetworkX)       (Obj 2)
  extraction_agent.py         builds the graph from Obj 1's processed elements     (Obj 2)
  vector_index.py             local embedding-based retrieval path                  (Obj 2)
  routing_agent.py            graph / vector / hybrid strategy selection            (Obj 2)
  verification_agent.py       judges evidence sufficiency, drives the retry loop   (Obj 2)
  qa_pipeline.py               QA graph: route -> retrieve -> verify -> (retry|answer) (Obj 2)
  system.py                    facade: ingest() / save() / load() / ask()
  cli.py                        `python -m privgraphai.cli ingest|ask ...`
tests/
  test_objective1_dryrun.py    ingestion graph wiring, with fakes
  test_objective2_dryrun.py    extraction, routing, verification, QA retry loop,
                                 and regression tests built from real failures
                                 seen running against an actual 20+ page paper
conftest.py                    lets `pytest tests/` find the privgraphai package
requirements.txt
app.py                         Flask web UI backend (local-only, no CDN)
templates/index.html           web UI page
static/style.css, app.js       web UI styling and client-side logic
README.md                      this file
```

## Setup

1. **Install Ollama**, start it, and pull the three local models:
   ```bash
   ollama serve
   ollama pull qwen2.5:3b-instruct-q4_K_M   # text: extraction, routing, verification, answers
   ollama pull moondream                     # vision: figure captioning
   ollama pull nomic-embed-text              # embeddings (Objective 2's vector path)
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Docling pulls in PyTorch and its own layout/OCR models (~a few hundred MB
   on first run) — the heaviest install here. Flask itself is pure Python
   and lightweight by comparison.

## Running it — two ways

### Option A: the web UI (recommended for demos/viva)

```bash
python app.py
```
Open **http://127.0.0.1:5000**. Upload a PDF, run ingestion, then ask
questions — all with live-streaming output and a status badge, in the
browser. Fully local: no CDN, no external fonts/scripts, nothing leaves
your machine. See the UI's own notes inline in `app.py` and the page itself
for details on how that's guaranteed.

### Option B: the command line

```bash
# Build the knowledge graph + vector index from a document:
python -m privgraphai.cli ingest path/to/document.pdf --graph graph.json --elements elements.json

# Ask a question against it:
python -m privgraphai.cli ask "How does the routing agent decide between graph and vector retrieval?" \
    --graph graph.json --elements elements.json
```

Both the UI and the CLI call the exact same underlying pipeline — the UI is
just a browser front end over the same `privgraphai.cli` commands.

## How Objective 2 actually works

**Knowledge graph construction** (`extraction_agent.py` + `graph_store.py`):
for every element Objective 1 produced, a local LLM call extracts named
entities and relations as JSON, written into a `KnowledgeGraph` (a NetworkX
`MultiDiGraph`). Entities and elements are nodes; `MENTIONED_IN` edges
connect an entity to the elements it appears in; labeled edges connect
entities to each other. Elements shorter than `config.min_extraction_chars`
(15 by default — page numbers, bare headers) skip extraction entirely to
avoid wasting LLM calls on large documents.

**Routing agent** (`routing_agent.py`): classifies a question as needing
`"graph"` (multi-hop, comparisons), `"vector"` (single-fact lookup), or
`"hybrid"` retrieval — with a keyword-heuristic fallback if the model's
JSON doesn't parse.

**Retrieval** (`qa_pipeline.py::_retrieve_node`): `"graph"` does a
breadth-first walk outward from entities matching the question, up to
`config.graph_max_hops`. `"vector"` does embedding-similarity search
(`nomic-embed-text`, cosine similarity). `"hybrid"` does both and merges.

**Evidence-verification agent** (`verification_agent.py`): the LLM judges
whether retrieved evidence is sufficient. If not, the graph loops back to
`retrieve` — widened to `"hybrid"` — for up to `config.qa_max_retries`
attempts, then answers with whatever evidence it has rather than looping
forever. This loop is the literal implementation of the project abstract's
line: *"an evidence-verification agent checks the relevance and adequacy of
retrieved information and initiates further retrieval when necessary."*

## Hardening from real-world testing

This isn't just unit-tested against synthetic cases — running Objective 2
against an actual 20+ page research paper (743 extracted elements) surfaced
real failure modes from the local 3B model under load, which are now
handled and specifically regression-tested in `test_objective2_dryrun.py`:

- **Malformed JSON** (the model forgets to close a bracket, e.g. an
  unclosed `"relations"` array) — `extraction_agent.py` now does bracket-aware
  repair, tested against the exact failing strings observed.
- **Concatenated JSON** (the model emits an empty `{}` "false start" then
  the real answer) — the parser finds every balanced top-level object and
  prefers the last one that actually matches the expected shape.
- **String vs. boolean `"sufficient"` values** — a small model occasionally
  returns `"sufficient": "false"` as a JSON *string*; since `bool("false")`
  is `True` in Python, a naive cast would silently invert this.
  `verification_agent.py` coerces both forms correctly.
- **CLI flag placement** — `-v`/`--verbose` is accepted both before and
  after the `ingest`/`ask` subcommand.

## Running the tests

```bash
pip install pytest
pytest tests/ -v      # 17 tests total
```

All 17 pass without Docling installed or an Ollama server running — the
dry-run tests inject fake/scripted clients so the orchestration logic
itself (graph wiring, the QA retry loop, JSON parsing/repair) is verified
independently of the heavy runtime dependencies.

## What's intentionally out of scope here

Baseline comparison systems (on-device Vector RAG, cloud-hosted GraphRAG)
and the comparative evaluation itself are Objectives 3 and 4 — separate
work, not part of this package.
