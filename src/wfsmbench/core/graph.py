"""Labeled, weighted, undirected simple graph — the shared data model.

The same type represents both host graphs (a database graph, with edge weights) and
pattern graphs (a candidate subgraph, with weights left as ``None``). Weights live on
host graphs only; patterns carry structure and labels, and their weights are read from
the host during matching (weights are used in scoring, not in
matching).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import networkx as nx

# An edge input is (u, v, edge_label) or (u, v, edge_label, weight).
EdgeInput = tuple


@dataclass
class Graph:
    node_labels: dict[int, int]                          # node id -> integer label
    edges: dict[tuple[int, int], tuple[int, Optional[float]]]  # (u<v) -> (edge_label, weight)
    graph_id: Optional[int] = None

    @staticmethod
    def of(node_labels: dict[int, int],
           edges: Iterable[EdgeInput],
           graph_id: Optional[int] = None) -> "Graph":
        labels = dict(node_labels)
        edge_map: dict[tuple[int, int], tuple[int, Optional[float]]] = {}
        for e in edges:
            u, v, edge_label = e[0], e[1], e[2]
            weight = e[3] if len(e) > 3 else None
            if u == v:
                raise ValueError(f"self-loop is not allowed: {e}")
            if u not in labels or v not in labels:
                raise ValueError(f"edge endpoint missing from node set: {e}")
            key = (u, v) if u < v else (v, u)
            edge_map[key] = (edge_label, weight)
        return Graph(labels, edge_map, graph_id)

    def nodes(self) -> list[int]:
        return list(self.node_labels)

    def num_nodes(self) -> int:
        return len(self.node_labels)

    def num_edges(self) -> int:
        return len(self.edges)

    def label(self, node: int) -> int:
        return self.node_labels[node]

    def neighbors(self, node: int) -> list[int]:
        out = []
        for (a, b) in self.edges:
            if a == node:
                out.append(b)
            elif b == node:
                out.append(a)
        return out

    def degree(self, node: int) -> int:
        return len(self.neighbors(node))

    def edge(self, u: int, v: int) -> Optional[tuple[int, Optional[float]]]:
        key = (u, v) if u < v else (v, u)
        return self.edges.get(key)

    def edge_label(self, u: int, v: int) -> Optional[int]:
        found = self.edge(u, v)
        return None if found is None else found[0]

    def weight(self, u: int, v: int) -> Optional[float]:
        found = self.edge(u, v)
        return None if found is None else found[1]

    def edge_items(self) -> Iterator[tuple[tuple[int, int], int]]:
        """Yield ((u, v), edge_label) with u < v; weights are omitted."""
        for (u, v), (edge_label, _weight) in self.edges.items():
            yield (u, v), edge_label

    def to_networkx(self) -> nx.Graph:
        """Instantiate the executable graph used for matching and evaluation."""
        g = nx.Graph()
        for node, lab in self.node_labels.items():
            g.add_node(node, label=lab)
        for (u, v), (edge_label, weight) in self.edges.items():
            g.add_edge(u, v, label=edge_label, weight=weight)
        return g
