"""Graph-edit moves — the discrete "velocity" alphabet of the search (proposal 4.5,
decision D1).

A particle changes through a sequence of moves. Moves are expressed at the structural
(template) level rather than at the node-id level, so the difference between two
particles is well defined even though their node ids are arbitrary:

- ``Grow(t)``    — add one edge of template ``t`` (close an edge between existing
                   compatible nodes if possible, otherwise expand outward from the pivot);
- ``Shrink(t)``  — remove one edge of template ``t`` (dropping any node it isolates);
- ``Repivot(l)`` — move the pivot to a node with label ``l``.

``diff`` (the operator delta of D1) reads off the moves that turn one particle into another
from the multiset difference of their edge templates plus a pivot change. ``apply_velocity``
(the operator phi) applies a sequence, skipping any move that cannot be applied or that would
break validity, so it always returns a valid particle. Moves are pure: each returns a new
particle and never mutates its input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..preprocess.template import EdgeTemplate


@dataclass(frozen=True)
class Grow:
    template: EdgeTemplate


@dataclass(frozen=True)
class Shrink:
    template: EdgeTemplate


@dataclass(frozen=True)
class Repivot:
    label: int


Move = Grow | Shrink | Repivot


# --- structural helpers -------------------------------------------------------

def _template_counter(pattern: Graph) -> Counter:
    counts: Counter = Counter()
    for (u, v), edge_label in pattern.edge_items():
        counts[EdgeTemplate.of(pattern.label(u), pattern.label(v), edge_label)] += 1
    return counts


def _new_node_id(pattern: Graph) -> int:
    return max(pattern.node_labels) + 1 if pattern.node_labels else 0


def connected(pattern: Graph) -> bool:
    nodes = pattern.nodes()
    if not nodes:
        return False
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        current = stack.pop()
        for neighbor in pattern.neighbors(current):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(nodes)


def _with_edge(particle: Particle, a: int, b: int, edge_label: int, pivot: int) -> Particle:
    pattern = particle.pattern
    edges = dict(pattern.edges)
    edges[(a, b) if a < b else (b, a)] = (edge_label, None)
    return Particle(Graph(dict(pattern.node_labels), edges, pattern.graph_id), pivot)


def _with_new_leaf(particle: Particle, anchor: int, new_label: int, edge_label: int) -> Particle:
    pattern = particle.pattern
    new_id = _new_node_id(pattern)
    node_labels = dict(pattern.node_labels)
    node_labels[new_id] = new_label
    edges = dict(pattern.edges)
    edges[(anchor, new_id) if anchor < new_id else (new_id, anchor)] = (edge_label, None)
    return Particle(Graph(node_labels, edges, pattern.graph_id), particle.pivot)


def _other_end(label: int, template: EdgeTemplate) -> int | None:
    if label == template.low_label:
        return template.high_label
    if label == template.high_label:
        return template.low_label
    return None


# --- move application ---------------------------------------------------------

def _apply_grow(particle: Particle, template: EdgeTemplate) -> Particle | None:
    pattern = particle.pattern
    nodes = sorted(pattern.node_labels)
    # 1) Close an edge between two existing, non-adjacent, label-compatible nodes.
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if pattern.edge(a, b) is not None:
                continue
            if EdgeTemplate.of(pattern.label(a), pattern.label(b), template.edge_label) == template:
                return _with_edge(particle, a, b, template.edge_label, particle.pivot)
    # 2) Expand outward from the pivot.
    other = _other_end(pattern.label(particle.pivot), template)
    if other is not None:
        return _with_new_leaf(particle, particle.pivot, other, template.edge_label)
    # 3) Expand from the lowest-id node compatible with the template.
    for node in nodes:
        other = _other_end(pattern.label(node), template)
        if other is not None:
            return _with_new_leaf(particle, node, other, template.edge_label)
    return None


def _apply_shrink(particle: Particle, template: EdgeTemplate) -> Particle | None:
    pattern = particle.pattern
    if pattern.num_edges() <= 1:
        return None
    for (u, v), edge_label in sorted(pattern.edge_items()):
        if EdgeTemplate.of(pattern.label(u), pattern.label(v), edge_label) != template:
            continue
        edges = dict(pattern.edges)
        del edges[(u, v)]
        node_labels = dict(pattern.node_labels)
        trimmed = Graph(node_labels, edges, pattern.graph_id)
        for isolated in [n for n in list(node_labels) if trimmed.degree(n) == 0]:
            del node_labels[isolated]
        pivot = particle.pivot if particle.pivot in node_labels else (
            min(node_labels) if node_labels else None)
        if pivot is None:
            continue
        candidate = Particle(Graph(node_labels, edges, pattern.graph_id), pivot)
        if connected(candidate.pattern):
            return candidate
    return None


def _apply_repivot(particle: Particle, label: int) -> Particle | None:
    pattern = particle.pattern
    for node in sorted(pattern.node_labels):
        if pattern.label(node) == label:
            return None if node == particle.pivot else Particle(pattern, node)
    return None


def apply_move(particle: Particle, move: Move) -> Particle | None:
    if isinstance(move, Grow):
        return _apply_grow(particle, move.template)
    if isinstance(move, Shrink):
        return _apply_shrink(particle, move.template)
    if isinstance(move, Repivot):
        return _apply_repivot(particle, move.label)
    raise TypeError(f"unknown move: {move!r}")


# --- Delta (diff) and Phi (apply) --------------------------------------------

def diff(source: Particle, target: Particle) -> list[Move]:
    """delta(x, y): the moves that turn ``source`` into a structure like ``target``."""
    source_templates = _template_counter(source.pattern)
    target_templates = _template_counter(target.pattern)
    moves: list[Move] = []
    for template in sorted(target_templates):
        for _ in range(target_templates[template] - source_templates.get(template, 0)):
            moves.append(Grow(template))
    for template in sorted(source_templates):
        for _ in range(source_templates[template] - target_templates.get(template, 0)):
            moves.append(Shrink(template))
    source_pivot_label = source.pattern.label(source.pivot)
    target_pivot_label = target.pattern.label(target.pivot)
    if source_pivot_label != target_pivot_label:
        moves.append(Repivot(target_pivot_label))
    return moves


def is_valid(particle: Particle, frequent_templates: set[EdgeTemplate], params: Params) -> bool:
    """Validity predicate (D1): connected, within size bounds, every edge frequent."""
    pattern = particle.pattern
    if pattern.num_edges() < 1:
        return False
    if pattern.num_nodes() > params.max_nodes or pattern.num_edges() > params.max_edges:
        return False
    if particle.pivot not in pattern.node_labels:
        return False
    if not connected(pattern):
        return False
    for (u, v), edge_label in pattern.edge_items():
        if EdgeTemplate.of(pattern.label(u), pattern.label(v), edge_label) not in frequent_templates:
            return False
    return True


def apply_velocity(particle: Particle,
                   velocity: list[Move],
                   frequent_templates: set[EdgeTemplate],
                   params: Params) -> Particle:
    """phi(x, v): apply the moves in order, keeping only those that stay valid."""
    current = particle
    for move in velocity:
        candidate = apply_move(current, move)
        if candidate is not None and is_valid(candidate, frequent_templates, params):
            current = candidate
    return current
