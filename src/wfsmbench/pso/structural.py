"""Structure-aware positions for the swarm: alignment-based edit moves.

The velocity of decision D1 compares particles by their edge-template *multiset*, which
throws the topology away. A path A-B-A-B and a star with three A leaves use the same three
templates, so the difference between them holds no structural move and no particle can be
pulled from one to the other. Enumerating BZR_MD at four edges, that projection maps 5939
distinct patterns onto 3003 positions -- one of them hiding nine patterns -- and the collapse
grows with pattern size, the same direction in which the swarm's gap to the optimum grows.

Making the difference structure-aware needs care. Describing a wanted edge by the degrees its
endpoints have *in the target* produces instructions that cannot be carried out: a one-edge
target says "an edge between two degree-1 nodes", which in a connected source would require
attaching to a degree-0 node. Measured, every such move failed -- 100% of them -- and the
swarm degenerated into a random walk.

So a move has to be phrased relative to the *source*. This module first aligns the two
patterns -- a label-respecting injective map from source nodes to target nodes chosen to
cover as many edges as possible -- and then reads the difference off that alignment:

- a target edge whose endpoints are both aligned, but which is missing in the source,
  becomes "close an edge between the source nodes playing those two roles";
- a target edge reaching an unaligned target node becomes "hang a new leaf off the source
  node playing that role";
- a source edge with no counterpart becomes "remove the edge in this position".

Positions are named by (label, degree) in the source, never by node id, so a velocity stays
meaningful when it is carried across iterations as inertia -- which is what D1 required of
the template-level moves in the first place.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..preprocess.template import EdgeTemplate
from .edits import Repivot, connected, _new_node_id

End = tuple[int, int]                    # (node label, node degree in its own pattern)
Context = tuple[End, End, int]           # (low end, high end, edge label)

# Alignment is brute-forced over label-compatible injections. Patterns are bounded by
# max_nodes (six by default), so this stays small; the cap only guards pathological cases.
_MAX_ALIGNMENTS = 20_000


@dataclass(frozen=True)
class GrowFrom:
    """Add an edge at the source position ``anchor``.

    ``close`` connects two nodes already present; otherwise a new leaf is attached.
    """
    anchor: End
    other_label: int
    edge_label: int
    close: bool


@dataclass(frozen=True)
class ShrinkEdge:
    """Remove the edge sitting at this source context."""
    context: Context


StructuralMove = GrowFrom | ShrinkEdge | Repivot


def degrees(pattern: Graph) -> Counter:
    counts: Counter = Counter()
    for (u, v) in pattern.edges:
        counts[u] += 1
        counts[v] += 1
    return counts


def context_of(pattern: Graph, u: int, v: int, edge_label: int,
               degree: Counter | None = None) -> Context:
    """The motif context of one edge; ends are ordered so the descriptor is canonical."""
    degree = degree if degree is not None else degrees(pattern)
    a: End = (pattern.label(u), degree[u])
    b: End = (pattern.label(v), degree[v])
    low, high = (a, b) if a <= b else (b, a)
    return (low, high, edge_label)


def context_counter(pattern: Graph) -> Counter:
    """The multiset of edge contexts -- the structure-aware position of a pattern."""
    degree = degrees(pattern)
    return Counter(context_of(pattern, u, v, edge_label, degree)
                   for (u, v), edge_label in pattern.edge_items())


def align(source: Graph, target: Graph) -> dict[int, int]:
    """A label-respecting *partial* injective map source->target covering as many edges as
    possible.

    Partial is the important word. Requiring every source node to find a partner makes the
    map empty whenever the two patterns use different labels, and an empty map yields no
    growth moves at all -- the particle can then never leave its seed. Instead each label is
    matched independently, up to however many nodes the two patterns share of it, and the
    assignment within a label is chosen to cover the most edges. Ties break on the
    lexicographically smallest mapping, so the result is deterministic.
    """
    if not source.node_labels or not target.node_labels:
        return {}

    by_label_source: dict[int, list[int]] = {}
    by_label_target: dict[int, list[int]] = {}
    for node, label in sorted(source.node_labels.items()):
        by_label_source.setdefault(label, []).append(node)
    for node, label in sorted(target.node_labels.items()):
        by_label_target.setdefault(label, []).append(node)

    shared = sorted(set(by_label_source) & set(by_label_target))
    if not shared:
        return {}

    target_edges = {frozenset(edge): label for edge, label in target.edge_items()}

    def score(mapping: dict[int, int]) -> int:
        return sum(1 for (u, v), label in source.edge_items()
                   if u in mapping and v in mapping
                   and target_edges.get(frozenset((mapping[u], mapping[v]))) == label)

    # Per label, try the ways of pairing its source nodes with its target nodes; combine the
    # best choice for each label. Label classes are tiny, so this stays cheap.
    mapping: dict[int, int] = {}
    for label in shared:
        sources = by_label_source[label]
        targets = by_label_target[label]
        width = min(len(sources), len(targets))
        best_local, best_score = None, -1
        examined = 0
        for chosen in permutations(targets, width):
            examined += 1
            if examined > _MAX_ALIGNMENTS:
                break
            candidate = dict(zip(sources[:width], chosen))
            value = score({**mapping, **candidate})
            if value > best_score:
                best_score, best_local = value, candidate
        if best_local:
            mapping.update(best_local)
    return mapping


def diff_aligned(source: Particle, target: Particle) -> list[StructuralMove]:
    """Delta read off an alignment, phrased in source-relative positions."""
    s_pattern, t_pattern = source.pattern, target.pattern
    mapping = align(s_pattern, t_pattern)
    inverse = {t: s for s, t in mapping.items()}
    s_degree = degrees(s_pattern)

    moves: list[StructuralMove] = []

    # Source edges with no counterpart in the target come out.
    target_edges = {frozenset(edge): label for edge, label in t_pattern.edge_items()}
    for (u, v), edge_label in s_pattern.edge_items():
        counterpart = None
        if u in mapping and v in mapping:
            counterpart = target_edges.get(frozenset((mapping[u], mapping[v])))
        if counterpart != edge_label:
            moves.append(ShrinkEdge(context_of(s_pattern, u, v, edge_label, s_degree)))

    # Target edges the source lacks go in, anchored at whichever endpoint the source has.
    source_edges = {frozenset(edge) for edge in s_pattern.edges}
    for (a, b), edge_label in t_pattern.edge_items():
        sa, sb = inverse.get(a), inverse.get(b)
        if sa is not None and sb is not None:
            if frozenset((sa, sb)) in source_edges:
                continue
            moves.append(GrowFrom((s_pattern.label(sa), s_degree[sa]),
                                  s_pattern.label(sb), edge_label, close=True))
        elif sa is not None or sb is not None:
            anchor_source = sa if sa is not None else sb
            leaf_label = t_pattern.label(b if sa is not None else a)
            moves.append(GrowFrom((s_pattern.label(anchor_source), s_degree[anchor_source]),
                                  leaf_label, edge_label, close=False))
        else:
            # Neither endpoint has a counterpart, so there is no position to copy. Grow from
            # the pivot instead, which is the anchor the edit alphabet falls back on anyway.
            moves.append(GrowFrom((s_pattern.label(source.pivot), s_degree[source.pivot]),
                                  t_pattern.label(a), edge_label, close=False))

    if s_pattern.label(source.pivot) != t_pattern.label(target.pivot):
        moves.append(Repivot(t_pattern.label(target.pivot)))
    return moves


def _apply_grow_from(particle: Particle, move: GrowFrom) -> Particle | None:
    pattern = particle.pattern
    degree = degrees(pattern)
    anchor_label, anchor_degree = move.anchor

    anchors = [n for n in sorted(pattern.node_labels)
               if pattern.label(n) == anchor_label and degree[n] == anchor_degree]
    if not anchors:      # fall back to the label alone; the degree is a preference, not a law
        anchors = [n for n in sorted(pattern.node_labels) if pattern.label(n) == anchor_label]
    if not anchors:
        return None

    for u in anchors:
        if move.close:
            for v in sorted(pattern.node_labels):
                if v == u or pattern.label(v) != move.other_label:
                    continue
                if pattern.edge(u, v) is not None:
                    continue
                edges = dict(pattern.edges)
                edges[(u, v) if u < v else (v, u)] = (move.edge_label, None)
                return Particle(Graph(dict(pattern.node_labels), edges, pattern.graph_id),
                                particle.pivot)
        else:
            new_id = _new_node_id(pattern)
            node_labels = dict(pattern.node_labels)
            node_labels[new_id] = move.other_label
            edges = dict(pattern.edges)
            edges[(u, new_id) if u < new_id else (new_id, u)] = (move.edge_label, None)
            return Particle(Graph(node_labels, edges, pattern.graph_id), particle.pivot)
    return None


def _apply_shrink_edge(particle: Particle, context: Context) -> Particle | None:
    pattern = particle.pattern
    degree = degrees(pattern)
    for (u, v), edge_label in pattern.edge_items():
        if context_of(pattern, u, v, edge_label, degree) != context:
            continue
        edges = {key: value for key, value in pattern.edges.items() if key != (u, v)}
        if not edges:
            return None
        kept = {node for edge in edges for node in edge}
        if particle.pivot not in kept:
            return None                      # never orphan the pivot
        node_labels = {n: l for n, l in pattern.node_labels.items() if n in kept}
        reduced = Graph(node_labels, edges, pattern.graph_id)
        if not connected(reduced):
            return None
        return Particle(reduced, particle.pivot)
    return None


def apply_structural(particle: Particle, move: StructuralMove) -> Particle | None:
    if isinstance(move, GrowFrom):
        return _apply_grow_from(particle, move)
    if isinstance(move, ShrinkEdge):
        return _apply_shrink_edge(particle, move.context)
    if isinstance(move, Repivot):
        for node, label in sorted(particle.pattern.node_labels.items()):
            if label == move.label:
                return Particle(particle.pattern, node)
        return None
    return None


def compose_structural(inertia: list[StructuralMove], cognitive: list[StructuralMove],
                       social: list[StructuralMove], max_velocity: int) -> list[StructuralMove]:
    """Omega for aligned moves: shrinks lead so grows have room, the pivot keeps its slot,
    and moves are kept in the order the parts contributed them."""
    combined = [*inertia, *cognitive, *social]
    shrinks = [m for m in combined if isinstance(m, ShrinkEdge)]
    grows = [m for m in combined if isinstance(m, GrowFrom)]
    seen: set = set()
    ordered: list[StructuralMove] = []
    for move in [*shrinks, *grows]:
        if move in seen:            # the same instruction twice adds nothing
            continue
        seen.add(move)
        ordered.append(move)

    repivots = [m for m in combined if isinstance(m, Repivot)]
    if repivots:
        ordered.insert(0, repivots[-1])
    return ordered[:max_velocity]


def apply_velocity_structural(particle: Particle, velocity: list[StructuralMove],
                              frequent: set[EdgeTemplate], params: Params) -> Particle:
    """Phi for aligned moves: apply in order, skipping anything that breaks validity."""
    from .edits import is_valid

    current = particle
    for move in velocity:
        candidate = apply_structural(current, move)
        if candidate is not None and is_valid(candidate, frequent, params):
            current = candidate
    return current


# The optimizer selects the difference operator by name; keep the old alias working.
diff_structural = diff_aligned
