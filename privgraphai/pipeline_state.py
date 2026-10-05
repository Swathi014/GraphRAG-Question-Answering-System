"""LangGraph state schema for PrivGraphAI (Objectives 1 & 2)."""

from __future__ import annotations

from typing import TypedDict

from .docling_parser import MultimodalElement


class PrivGraphAIState(TypedDict, total=False):
    source_path: str
    raw_elements: list[MultimodalElement]
    processed_elements: list[dict]
    errors: list[str]

    question: str
    strategy: str
    retrieved: list[dict]
    retry_count: int
    sufficient: bool
    missing: str
    answer: str
