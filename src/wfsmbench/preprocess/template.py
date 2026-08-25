"""Frequent edge template: t = (label(u), label(v), edge_label).

A template is the label signature of an undirected edge — two node labels and an edge
label — with the node labels kept in ``low <= high`` order so the two orientations of an
undirected edge map to the same template. Templates seed candidate generation and
constrain expansion to structures that actually occur in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class EdgeTemplate:
    low_label: int
    high_label: int
    edge_label: int

    @staticmethod
    def of(label_a: int, label_b: int, edge_label: int) -> "EdgeTemplate":
        low, high = (label_a, label_b) if label_a <= label_b else (label_b, label_a)
        return EdgeTemplate(low, high, edge_label)
