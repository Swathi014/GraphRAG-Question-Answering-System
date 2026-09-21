"""LangGraph state schema for PrivGraphAI (Objectives 1 & 2).

One TypedDict covers both graphs that use it:
  - the Objective 1 ingestion graph (source_path -> processed_elements)
  - the Objective 2 QA graph (question -> answer), which reuses this same
    state shape via its own subset of fields.

`total=False` means every field is optional in any given state dict — each
node only needs to touch the fields relevant to its own graph.
"""

from __future__ import annotations

from typing import TypedDict

from .docling_parser import MultimodalElement


class PrivGraphAIState(TypedDict, total=False):
    # -- Objective 1: ingestion --------------------------------------
    source_path: str
    raw_elements: list[MultimodalElement]
    processed_elements: list[dict]
    errors: list[str]

    # -- Objective 2: question answering -------------------------------
    question: str
    strategy: str            # "graph" | "vector" | "hybrid"
    retrieved: list[dict]
    retry_count: int
    sufficient: bool
    missing: str
    answer: str
