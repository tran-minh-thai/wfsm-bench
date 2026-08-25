"""Shared building blocks for the search baselines.

Every baseline optimizes the SAME objective as the PSO search (the fitness of
`pso.optimizer.evaluate`) and reuses the same pattern enumeration and archive, so the
comparison is fair. Only the search strategy differs.
"""

from __future__ import annotations

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..core.signature import canonical_signature
from ..preprocess.statistics import DatasetStats
from ..preprocess.template import EdgeTemplate
from ..pso.edits import is_valid
from ..pso.optimizer import Candidate, Result, evaluate, select_top_k

__all__ = [
    "Candidate", "Result", "evaluate", "select_top_k", "Params",
    "other_end", "as_particle", "seed_particles", "extensions", "valid_children",
    "pattern_templates", "Archive",
]


def other_end(label: int, template: EdgeTemplate) -> int | None:
    if label == template.low_label:
        return template.high_label
    if label == template.high_label:
        return template.low_label
    return None


def as_particle(pattern: Graph) -> Particle:
    return Particle(pattern, min(pattern.node_labels))  # pivot is irrelevant to scoring


def seed_particles(frequent: list[EdgeTemplate]) -> list[Particle]:
    return [as_particle(Graph.of(
        {0: t.low_label, 1: t.high_label}, [(0, 1, t.edge_label)]))
        for t in frequent]


def pattern_templates(pattern: Graph) -> list[EdgeTemplate]:
    return [EdgeTemplate.of(pattern.label(u), pattern.label(v), edge_label)
            for (u, v), edge_label in pattern.edge_items()]


def extensions(pattern: Graph, frequent: set[EdgeTemplate]) -> list[Graph]:
    """All single-edge extensions of ``pattern`` using frequent templates."""
    nodes = sorted(pattern.node_labels)
    children: list[Graph] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if pattern.edge(a, b) is not None:
                continue
            for template in frequent:
                if EdgeTemplate.of(pattern.label(a), pattern.label(b), template.edge_label) == template:
                    edges = dict(pattern.edges)
                    edges[(a, b)] = (template.edge_label, None)
                    children.append(Graph(dict(pattern.node_labels), edges, pattern.graph_id))
    new_id = (max(nodes) + 1) if nodes else 0
    for anchor in nodes:
        for template in frequent:
            other = other_end(pattern.label(anchor), template)
            if other is None:
                continue
            node_labels = dict(pattern.node_labels)
            node_labels[new_id] = other
            edges = dict(pattern.edges)
            edges[(anchor, new_id) if anchor < new_id else (new_id, anchor)] = (template.edge_label, None)
            children.append(Graph(node_labels, edges, pattern.graph_id))
    return children


def valid_children(particle: Particle, frequent: set[EdgeTemplate], params: Params) -> list[Particle]:
    return [as_particle(g) for g in extensions(particle.pattern, frequent)
            if is_valid(as_particle(g), frequent, params)]


class Archive:
    """Top-K by fitness, keyed by canonical signature, frequent patterns only."""

    def __init__(self, params: Params) -> None:
        self._params = params
        self._by_signature: dict[str, Candidate] = {}

    def add(self, candidate: Candidate) -> None:
        if candidate.evaluation.support < self._params.min_support:
            return
        signature = candidate.signature
        existing = self._by_signature.get(signature)
        if existing is None or candidate.fitness > existing.fitness:
            self._by_signature[signature] = candidate

    def candidates(self) -> list[Candidate]:
        """Everything admitted so far, not just the reported Top-K."""
        return list(self._by_signature.values())

    def result(self) -> Result:
        candidates = list(self._by_signature.values())
        best = max(candidates, key=lambda c: c.fitness, default=None)
        return Result(top_k=select_top_k(candidates, self._params), gbest=best,
                      pool=candidates)
