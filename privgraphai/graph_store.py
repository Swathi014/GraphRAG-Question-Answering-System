"""In-memory multimodal knowledge graph for PrivGraphAI — Objective 2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger("privgraphai.graph_store")


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
        node_id = f"entity::{name.strip().lower()}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, kind="entity", name=name.strip(), entity_type=entity_type)
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
        src_id = self.add_entity(source_entity, source_type)
        tgt_id = self.add_entity(target_entity, target_type)
        self.graph.add_edge(src_id, tgt_id, relation=relation_label)

    def find_entities(self, text: str) -> list[str]:
        needle = text.strip().lower()
        return [
            n
            for n, data in self.graph.nodes(data=True)
            if data.get("kind") == "entity" and data.get("name", "").lower() in needle
        ]

    def multi_hop_elements(self, seed_entity_ids: list[str], max_hops: int = 2) -> list[dict]:
        if not seed_entity_ids:
            return []

        visited_entities = set(seed_entity_ids)
        frontier = list(seed_entity_ids)
        collected: dict[str, dict] = {}

        for _ in range(max_hops):
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
                        collected[neighbor] = {"element_id": neighbor, **ndata}
                    elif ndata.get("kind") == "entity" and neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return list(collected.values())

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
