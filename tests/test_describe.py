"""Descriptive statistics, checked against hand-counted values on the sample collection."""

from pathlib import Path

import pytest

from wfsmbench.experiments import dataset_descriptive
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def test_dataset_descriptive_on_sample() -> None:
    graphs, _ = load_dataset(SAMPLE)
    described = dataset_descriptive(graphs)
    assert described["avg_nodes"] == pytest.approx(8 / 3)      # 3 + 3 + 2 nodes
    assert described["avg_edges"] == pytest.approx(5 / 3)      # 2 + 2 + 1 edges
    assert described["avg_weight"] == pytest.approx(0.68)      # (.8+.4+.6+.6+1.0)/5


def test_a_collection_with_no_weights_reports_zero_rather_than_dividing_by_zero() -> None:
    graphs, _ = load_dataset(SAMPLE)
    for graph in graphs:
        for edge, (label, _weight) in list(graph.edges.items()):
            graph.edges[edge] = (label, None)
    assert dataset_descriptive(graphs)["avg_weight"] == 0.0
