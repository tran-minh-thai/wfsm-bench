"""Pivot selection (proposal 4.3, eqs 11-17).

Each node of a pattern is scored by a weighted sum of four structural / statistical
terms, and the highest-scoring node becomes the pivot — the anchor that guides
expansion.

    S(v) = l1*f_deg(v) + l2*f_wdeg(v) + l3*f_cent(v) + l4*f_lab(v)

    f_deg(v)  = degree of v in the pattern
    f_wdeg(v) = sum of representative weights of v's incident edges (decision D8:
                each pattern edge takes its template's mean dataset weight)
    f_cent(v) = degree centrality, deg(v) / (|V| - 1)
    f_lab(v)  = fraction of dataset graphs containing v's node label

Ties are broken by the smaller node id.
"""

from __future__ import annotations

from ..config import Params
from ..core.graph import Graph
from ..preprocess.statistics import DatasetStats
from ..preprocess.template import EdgeTemplate


def _weighted_degree(pattern: Graph, node: int, stats: DatasetStats) -> float:
    total = 0.0
    for neighbor in pattern.neighbors(node):
        template = EdgeTemplate.of(
            pattern.label(node), pattern.label(neighbor),
            pattern.edge_label(node, neighbor),
        )
        total += stats.template_weight(template)
    return total


def pivot_score(pattern: Graph, node: int, stats: DatasetStats, params: Params) -> float:
    degree = pattern.degree(node)
    num_nodes = pattern.num_nodes()
    f_deg = float(degree)
    f_wdeg = _weighted_degree(pattern, node, stats)
    f_cent = degree / (num_nodes - 1) if num_nodes > 1 else 0.0
    f_lab = stats.label_frequency(pattern.label(node))
    return (
        params.lambda1 * f_deg
        + params.lambda2 * f_wdeg
        + params.lambda3 * f_cent
        + params.lambda4 * f_lab
    )


def select_pivot(pattern: Graph, stats: DatasetStats, params: Params, rng=None) -> int:
    """Select the pivot node under the configured strategy (for the ablation study):

    - ``scientific`` (default): the highest S(v);
    - ``degree``: the highest-degree node;
    - ``random``: a uniformly random node (needs ``rng``);
    - ``none``: no guidance — the lowest-id node.

    Ties break on the smaller node id.
    """
    nodes = pattern.nodes()
    if not nodes:
        raise ValueError("cannot select a pivot from an empty pattern")

    strategy = params.pivot_strategy
    if strategy == "none":
        return min(nodes)
    if strategy == "random":
        if rng is None:
            raise ValueError("the 'random' pivot strategy requires an rng")
        return int(rng.choice(sorted(nodes)))
    if strategy == "degree":
        return max(nodes, key=lambda v: (pattern.degree(v), -v))
    # scientific
    return max(nodes, key=lambda v: (pivot_score(pattern, v, stats, params), -v))
