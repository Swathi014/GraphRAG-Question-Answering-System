"""Central configuration for PrivGraphAI (Objectives 1 & 2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivGraphAIConfig:
    ollama_host: str = "http://localhost:11434"
    text_model: str = "qwen2.5:3b-instruct-q4_K_M"
    vision_model: str = "moondream"
    embedding_model: str = "nomic-embed-text"

    image_export_dir: str = "./_extracted_images"

    graph_max_hops: int = 2
    qa_max_retries: int = 2
    min_extraction_chars: int = 15

    request_timeout_s: int = 420


DEFAULT_CONFIG = PrivGraphAIConfig()
