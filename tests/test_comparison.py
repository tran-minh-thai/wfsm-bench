"""PSO vs exhaustive comparison harness, locked on the sample."""

from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.experiments import comparison_to_csv, run_comparison
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def sample_graphs():
    graphs, _ = load_dataset(SAMPLE)
    return graphs


def test_comparison_reports_gap_and_speedup():
    c = run_comparison(sample_graphs(), "sample", graph_limit=3, seeds=[0, 1, 2], params=Params())
    assert c.dataset == "sample" and c.graph_limit == 3 and c.num_seeds == 3
    assert c.exhaustive_fitness == pytest.approx(0.1833333, abs=1e-6)  # path A-B-C
    assert c.pso_fitness_mean <= c.exhaustive_fitness + 1e-9           # PSO never beats optimum
    assert c.fitness_gap_pct >= -1e-9                                   # gap is non-negative
    assert 0.0 <= c.pso_hits_optimum_rate <= 1.0
    assert c.num_frequent >= 1
    assert c.speedup > 0.0


def test_comparison_csv():
    rows = [run_comparison(sample_graphs(), "sample", 3, [0, 1], Params())]
    csv = comparison_to_csv(rows)
    header = csv.splitlines()[0]
    assert header.startswith("dataset,graph_limit,num_seeds,num_frequent,exhaustive_fitness")
