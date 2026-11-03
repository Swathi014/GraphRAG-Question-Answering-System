"""In-memory multimodal knowledge graph for PrivGraphAI - Objective 2.

PROPOSED version. Changes from the original:
  * add_entity / add_relation validate their inputs and raise ValueError for non-string or empty
    names (the original crashed with "'list' object has no attribute 'strip'" when the 3B model
    returned a list where a string belongs). The extraction agent now catches this per item.
  * find_entities matches whole words only and ignores very short names. Before, an entity named
    "ti" or "d" matched inside almost any question and pulled in unrelated elements.
  * multi_hop_elements returns elements ranked: closer to the question's entities first, and
    elements linked to several of them before elements linked to one.
Everything else (method names, saved file format) is unchanged, so old saved graphs still load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger("privgraphai.graph_store")

_MIN_ENTITY_CHARS = 3  # shorter entity names are too ambiguous to match inside a question


def _clean_name(name) -> Optional[str]:
    if not isinstance(name, str):
        return None
    name = " ".join(name.split())
    return name or None


def _contains_word(haystack: str, needle: str) -> bool:
    """True if `needle` occurs in `haystack` as a whole word or phrase (both lower-case)."""
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return False
        before_ok = i == 0 or not haystack[i - 1].isalnum()
        j = i + len(needle)
        after_ok = j == len(haystack) or not haystack[j].isalnum()
        if before_ok and after_ok:
            return True
        start = i + 1


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_element(
        self, element_id: str, element_type: str, content: str, page_number: Optional[int] = None
    ) -> None:
        self.graph.add_node(
            element_id, kind="element", element_type=element_type, content=content, page_number=page_number
        )

    def add_entity(self, name: str, entity_type: str = "unknown") -> str:
        clean = _clean_name(name)
        if clean is None:
            raise ValueError(f"entity name must be a non-empty string, got {name!r}")
        node_id = f"entity::{clean.lower()}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                kind="entity",
                name=clean,
                entity_type=entity_type if isinstance(entity_type, str) else "unknown",
            )
        return node_id

    def link_entity_to_element(self, entity_name: str, element_id: str, entity_type: str = "unknown") -> None:
        entity_id = self.add_entity(entity_name, entity_type)
        if self.graph.has_node(element_id):
            self.graph.add_edge(entity_id, element_id, relation="MENTIONED_IN")
        else:
            logger.warning("link_entity_to_element: unknown element_id %s", element_id)

    def add_relation(
        self,
        source_entity: str,
        relation_label: str,
        target_entity: str,
        source_type: str = "unknown",
        target_type: str = "unknown",
    ) -> None:
        # Validate everything first, so a bad relation leaves no half-added nodes behind.
        if _clean_name(source_entity) is None or _clean_name(target_entity) is None:
            raise ValueError(f"relation endpoints must be non-empty strings, got {source_entity!r}, {target_entity!r}")
        label = _clean_name(relation_label)
        if label is None:
            raise ValueError(f"relation label must be a non-empty string, got {relation_label!r}")
        src_id = self.add_entity(source_entity, source_type)
        tgt_id = self.add_entity(target_entity, target_type)
        self.graph.add_edge(src_id, tgt_id, relation=label)

    def find_entities(self, text: str) -> list[str]:
        needle = " ".join(text.lower().split())
        found = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("kind") != "entity":
                continue
            name = str(data.get("name", "")).lower()
            if len(name) >= _MIN_ENTITY_CHARS and _contains_word(needle, name):
                found.append(node_id)
        return found

    def multi_hop_elements(self, seed_entity_ids: list[str], max_hops: int = 2) -> list[dict]:
        """Elements reachable from the seed entities within `max_hops`, best first.

        Each returned dict has the node data plus `hops` (1 = directly linked to a seed entity)
        and `seed_hits` (how many seed entities link to it directly).
        """
        if not seed_entity_ids:
            return []

        seeds = set(seed_entity_ids)
        visited_entities = set(seed_entity_ids)
        frontier = list(seed_entity_ids)
        collected: dict[str, dict] = {}

        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                if node_id not in self.graph:
                    continue
                neighbors = list(self.graph.out_edges(node_id, data=True)) + [
                    (target, source, data) for source, target, data in self.graph.in_edges(node_id, data=True)
                ]
                for _, neighbor, _data in neighbors:
                    ndata = self.graph.nodes[neighbor]
                    if ndata.get("kind") == "element":
                        entry = collected.setdefault(
                            neighbor, {"element_id": neighbor, **ndata, "hops": hop, "seed_hits": 0}
                        )
                        if node_id in seeds:
                            entry["seed_hits"] += 1
                    elif ndata.get("kind") == "entity" and neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return sorted(collected.values(), key=lambda d: (d["hops"], -d["seed_hits"]))

    def stats(self) -> dict:
        kinds = [d.get("kind") for _, d in self.graph.nodes(data=True)]
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_entities": kinds.count("entity"),
            "num_elements": kinds.count("element"),
        }

    def save(self, path: str | Path) -> None:
        data = nx.node_link_data(self.graph, edges="edges")
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Knowledge graph saved to %s (%s)", path, self.stats())

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        kg.graph = nx.node_link_graph(data, edges="edges", multigraph=True, directed=True)
        return kg
