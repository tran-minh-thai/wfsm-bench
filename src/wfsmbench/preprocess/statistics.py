"""Dataset-level statistics computed once, before the search.

Provides what the later phases need:
- ``label_frequency`` — the f_lab term of pivot scoring (fraction of graphs a node
  label appears in);
- ``frequent_edge_templates`` — the templates that seed and constrain candidate
  generation;
- ``max_edge_weight`` and the ``*_reference`` bounds — the fixed normalization anchors
  for the fitness function (decision D2, docs/DesignDecisions.md).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..core.graph import Graph
from .template import EdgeTemplate


@dataclass
class DatasetStats:
    num_graphs: int
    node_label_graph_freq: dict[int, int]              # node label -> number of graphs containing it
    edge_template_support: dict[EdgeTemplate, int]      # template -> number of graphs containing it
    edge_template_mean_weight: dict[EdgeTemplate, float]  # template -> mean weight over all instances
    max_edge_weight: float
    min_edge_weight: float

    def label_frequency(self, label: int) -> float:
        """f_lab: fraction of graphs that contain a node with this label."""
        if self.num_graphs == 0:
            return 0.0
        return self.node_label_graph_freq.get(label, 0) / self.num_graphs

    def template_weight(self, template: EdgeTemplate) -> float:
        """Representative weight of an edge of this template (decision D8): the mean
        weight the template takes across the dataset. 0.0 if the template is unseen."""
        return self.edge_template_mean_weight.get(template, 0.0)

    def frequent_edge_templates(self, min_support: int) -> list[EdgeTemplate]:
        """Templates occurring in at least ``min_support`` graphs, in canonical order."""
        return sorted(t for t, s in self.edge_template_support.items() if s >= min_support)

    def atw_reference(self, size_reference: int) -> float:
        """Upper anchor for ATW normalization: size_reference * max edge weight (D2).

        ``size_reference`` is a constant of the study (``Params.size_reference``), fixed once
        and held across every size setting. It is deliberately NOT the search bound
        ``max_edges``: an anchor that moves with the bound makes F(g) depend on the search
        configuration rather than on the pattern, which is the defect recorded in config.py.
        """
        return size_reference * self.max_edge_weight

    def atw_reference_discounted(self, size_reference: int, rho: float) -> float:
        """Upper anchor for ATW under weight_mode="discounted".

        The largest discounted sum a pattern of ``size_reference`` edges could reach, which is
        the geometric series W_max * (1 + rho + ... + rho^(R-1)). At rho = 1 it collapses to
        ``atw_reference``, so the two modes agree exactly where they should.
        """
        span = (size_reference if rho == 1.0
                else (1.0 - rho ** size_reference) / (1.0 - rho))
        return span * self.max_edge_weight

    def wsup_reference(self, size_reference: int,
                       num_graphs_ref: int | None = None) -> float:
        """Upper anchor for WSUP normalization: N_ref * ATW_ref (D2)."""
        n = self.num_graphs if num_graphs_ref is None else num_graphs_ref
        return n * self.atw_reference(size_reference)


def build_statistics(graphs: list[Graph]) -> DatasetStats:
    node_label_graphs: dict[int, set[int]] = defaultdict(set)
    template_graphs: dict[EdgeTemplate, set[int]] = defaultdict(set)
    template_weight_sum: dict[EdgeTemplate, float] = defaultdict(float)
    template_weight_count: dict[EdgeTemplate, int] = defaultdict(int)
    max_w = float("-inf")
    min_w = float("inf")

    for gi, g in enumerate(graphs):
        for label in g.node_labels.values():
            node_label_graphs[label].add(gi)
        for (u, v), (edge_label, weight) in g.edges.items():
            template = EdgeTemplate.of(g.label(u), g.label(v), edge_label)
            template_graphs[template].add(gi)
            if weight is not None:
                template_weight_sum[template] += weight
                template_weight_count[template] += 1
                max_w = max(max_w, weight)
                min_w = min(min_w, weight)

    if max_w == float("-inf"):  # no weighted edges anywhere
        max_w = min_w = 0.0

    return DatasetStats(
        num_graphs=len(graphs),
        node_label_graph_freq={lab: len(gs) for lab, gs in node_label_graphs.items()},
        edge_template_support={t: len(gs) for t, gs in template_graphs.items()},
        edge_template_mean_weight={
            t: template_weight_sum[t] / template_weight_count[t]
            for t in template_weight_count
        },
        max_edge_weight=max_w,
        min_edge_weight=min_w,
    )
