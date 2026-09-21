"""
PrivGraphAI — Objectives 1 & 2.

Objective 1 (privgraphai.pipeline.PrivGraphAIPipeline):
    Docling extraction -> Ollama processing -> LangGraph orchestration,
    turning a raw document into a list of processed multimodal elements.

Objective 2 (this package's new modules):
    - graph_store.KnowledgeGraph       in-memory multimodal knowledge graph
    - extraction_agent.ExtractionAgent builds the graph from Obj 1's output
    - vector_index.SimpleVectorIndex   local embedding-based retrieval path
    - routing_agent.RoutingAgent       picks graph / vector / hybrid retrieval
    - verification_agent.VerificationAgent  judges evidence sufficiency
    - qa_pipeline.QAPipeline           LangGraph loop: route -> retrieve ->
                                        verify -> (retry | answer)
    - system.PrivGraphAISystem         facade tying both objectives together
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

__version__ = "0.2.0"
