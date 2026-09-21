"""Central configuration for PrivGraphAI (Objectives 1 & 2).

Model choices are deliberately small, CPU-friendly quantized models —
matching the target hardware profile (a consumer laptop with no dedicated
GPU, e.g. AMD Ryzen 5 5600H / 16GB RAM). Swap them for larger quantizations
if you're running on a machine with a discrete GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivGraphAIConfig:
    # -- Local model serving (Ollama) ------------------------------------
    ollama_host: str = "http://localhost:11434"
    text_model: str = "qwen2.5:3b-instruct-q4_K_M"   # quantized LLM: extraction/routing/verification/answers
    vision_model: str = "moondream"                   # quantized VLM: figure captioning (Objective 1)
    embedding_model: str = "nomic-embed-text"          # local embedding model (Objective 2's vector path)

    # -- Docling extraction ----------------------------------------------
    image_export_dir: str = "./_extracted_images"

    # -- Knowledge graph & retrieval (Objective 2) ------------------------
    graph_max_hops: int = 2       # how far multi-hop retrieval walks from a seed entity
    qa_max_retries: int = 2       # bounded re-retrieval loop before answering with best-effort evidence

    # -- Runtime -----------------------------------------------------------
    request_timeout_s: int = 300


DEFAULT_CONFIG = PrivGraphAIConfig()
