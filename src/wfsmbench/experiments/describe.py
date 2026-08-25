"""Descriptive statistics of a graph collection.

This is what remains of a figure module that drew ten charts. Those charts read columns --
`exhaustive_fitness`, `pso_runtime_mean`, `diversity_mean`, `pivot_strategy` -- that appear
in **zero** of the current result files, because the generation that produced them was
removed in the audit. Figures for the paper come from `scripts/make_figures.py`, which owns
its own matplotlib and reads the artefacts that still exist.

Ten functions that can only run on data nobody has are worse than no functions at all: a
later run would call one, get a chart, and have no way to tell that its inputs were retired.
The descriptive statistic below is kept because it computes a real quantity that
`docs/DATASETS.md` quotes, and because it does not depend on any CSV.
"""

from __future__ import annotations

from ..core.graph import Graph


def dataset_descriptive(graphs: list[Graph]) -> dict[str, float]:
    """Mean nodes, mean edges, and mean edge weight over a collection."""
    n = len(graphs)
    avg_nodes = sum(g.num_nodes() for g in graphs) / n
    avg_edges = sum(g.num_edges() for g in graphs) / n
    weights = [w for g in graphs for (_e, (_l, w)) in g.edges.items() if w is not None]
    avg_weight = sum(weights) / len(weights) if weights else 0.0
    return {"avg_nodes": avg_nodes, "avg_edges": avg_edges, "avg_weight": avg_weight}
