"""Diversity of a result set.

Fitness measures how good the single best pattern is; diversity measures how varied the
whole Top-K set is. A greedy trajectory tends to return near-duplicates of one strong
structure, whereas a population search can return several structurally different high-
quality patterns. We quantify this with the mean pairwise structural distance over the
Top-K, where the distance between two patterns is the Jaccard distance of their edge-
template sets (0 = identical template content, 1 = disjoint).
"""

from __future__ import annotations

from itertools import combinations

from ..core.graph import Graph
from ..preprocess.template import EdgeTemplate


def _template_set(pattern: Graph) -> frozenset[EdgeTemplate]:
    return frozenset(
        EdgeTemplate.of(pattern.label(u), pattern.label(v), edge_label)
        for (u, v), edge_label in pattern.edge_items()
    )


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


def mean_pairwise_diversity(patterns: list[Graph]) -> float:
    """Average template-Jaccard distance over all pairs; 0.0 for fewer than two patterns."""
    sets = [_template_set(p) for p in patterns]
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return 0.0
    return sum(_jaccard_distance(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def distinct_sizes(patterns: list[Graph]) -> int:
    """Number of distinct (node count, edge count) shapes in the set."""
    return len({(p.num_nodes(), p.num_edges()) for p in patterns})


def select_diverse(scored: list[tuple[float, Graph]], k: int,
                   weight: float) -> list[int]:
    """Pick k indices of ``scored`` trading fitness off against structural novelty.

    ``weight`` is the trade-off mu: 0.0 selects the plain top-k by fitness, 1.0 selects
    purely for novelty. In between, each pick maximizes

        (1 - mu) * fitness_norm + mu * (distance to the nearest already-picked pattern)

    Fitness is min-max normalized over the candidate pool so both terms share the [0,1]
    scale; the first pick is always the fittest candidate. Ties break on the lower index,
    keeping selection deterministic for a fixed candidate order.
    """
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
    if weight <= 0.0 or len(order) <= 1:
        return order[:k]

    values = [f for f, _ in scored]
    low, high = min(values), max(values)
    span = high - low
    normalized = [1.0 if span == 0 else (f - low) / span for f in values]
    sets = [_template_set(p) for _, p in scored]

    selected = [order[0]]
    nearest = {i: _jaccard_distance(sets[i], sets[selected[0]]) for i in order[1:]}
    while len(selected) < min(k, len(order)):
        pick = max(nearest, key=lambda i: (
            (1.0 - weight) * normalized[i] + weight * nearest[i], -i))
        selected.append(pick)
        del nearest[pick]
        for i in nearest:
            nearest[i] = min(nearest[i], _jaccard_distance(sets[i], sets[pick]))
    return selected
