"""Command-line entry point for PrivGraphAI (Objectives 1 & 2).

Usage:
    # Build the knowledge graph + vector index from a document:
    python -m privgraphai.cli ingest path/to/document.pdf --graph graph.json --elements elements.json

    # Ask a question against a previously built graph:
    python -m privgraphai.cli ask "How does the routing agent decide between graph and vector retrieval?" \\
        --graph graph.json --elements elements.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from .config import DEFAULT_CONFIG
from .system import PrivGraphAISystem


def _cmd_ingest(args: argparse.Namespace) -> int:
    system = PrivGraphAISystem(DEFAULT_CONFIG)
    result = system.ingest(args.source)
    system.save(args.graph, args.elements)

    print(json.dumps(result, indent=2, default=str))
    n_err = len(result.get("ingestion_errors", []))
    print(f"\nKnowledge graph saved to {args.graph}" + (f", elements to {args.elements}" if args.elements else ""))
    return 0 if n_err == 0 else 1


def _cmd_ask(args: argparse.Namespace) -> int:
    system = PrivGraphAISystem(DEFAULT_CONFIG)
    system.load(args.graph, args.elements)
    result = system.ask(args.question)

    print(f"\nStrategy used : {result.get('strategy')}")
    print(f"Retries       : {result.get('retry_count', 0) - 1}")
    print(f"Evidence found: {len(result.get('retrieved', []))} element(s)")
    print(f"Sufficient?   : {result.get('sufficient')}")
    print(f"\nAnswer:\n{result.get('answer', '(no answer generated)')}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PrivGraphAI — on-device Agentic GraphRAG (Objectives 1 & 2)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse a document and build its knowledge graph")
    p_ingest.add_argument("source", help="Path to a PDF (or other Docling-supported) document")
    p_ingest.add_argument("--graph", default="graph.json", help="Where to save the knowledge graph")
    p_ingest.add_argument("--elements", default="elements.json", help="Where to save the processed elements")
    p_ingest.add_argument("-v", "--verbose", action="store_true")  # also accepted after the subcommand
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against a previously built knowledge graph")
    p_ask.add_argument("question")
    p_ask.add_argument("--graph", default="graph.json", help="Path to a saved knowledge graph")
    p_ask.add_argument("--elements", default="elements.json", help="Path to saved processed elements")
    p_ask.add_argument("-v", "--verbose", action="store_true")  # also accepted after the subcommand
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
