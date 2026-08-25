"""All-methods comparison harness, locked on the sample."""

from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.experiments import (
    method_comparison_to_csv,
    run_method_comparison,
)
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def sample_graphs():
    graphs, _ = load_dataset(SAMPLE)
    return graphs


def test_method_comparison_covers_all_methods_and_bounds():
    rows = run_method_comparison(sample_graphs(), "sample", 3, [0, 1], Params())
    by_method = {r.method: r for r in rows}
    assert set(by_method) == {"exhaustive", "greedy", "pso", "hybrid", "random", "genetic"}
    optimum = by_method["exhaustive"].fitness_mean
    assert by_method["exhaustive"].gap_pct == pytest.approx(0.0)
    for method, r in by_method.items():
        assert r.fitness_mean <= optimum + 1e-9          # nothing beats the optimum
        assert r.gap_pct >= -1e-9


def test_method_comparison_csv():
    rows = run_method_comparison(sample_graphs(), "sample", 3, [0, 1], Params())
    csv = method_comparison_to_csv(rows)
    assert csv.splitlines()[0].startswith("dataset,graph_limit,method,num_seeds,fitness_mean")
