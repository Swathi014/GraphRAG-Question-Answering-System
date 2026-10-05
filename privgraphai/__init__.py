"""
PrivGraphAI — Objectives 1 & 2.
"""

from .config import DEFAULT_CONFIG, PrivGraphAIConfig
from .docling_parser import DoclingParser, MultimodalElement
from .extraction_agent import ExtractionAgent
from .graph_store import KnowledgeGraph
from .ollama_client import OllamaClient
from .pipeline import PrivGraphAIPipeline
from .qa_pipeline import QAPipeline
from .routing_agent import RoutingAgent
from .system import PrivGraphAISystem
from .vector_index import SimpleVectorIndex
from .verification_agent import VerificationAgent

__all__ = [
    "DEFAULT_CONFIG",
    "PrivGraphAIConfig",
    "DoclingParser",
    "MultimodalElement",
    "OllamaClient",
    "PrivGraphAIPipeline",
    "KnowledgeGraph",
    "ExtractionAgent",
    "SimpleVectorIndex",
    "RoutingAgent",
    "VerificationAgent",
    "QAPipeline",
    "PrivGraphAISystem",
]

__version__ = "0.3.0"
