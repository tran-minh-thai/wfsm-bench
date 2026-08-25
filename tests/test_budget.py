"""PSO budget sweep against a fixed exhaustive optimum, locked on the sample."""

from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.experiments import budget_to_csv, run_budget_sweep
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def sample_graphs():
    graphs, _ = load_dataset(SAMPLE)
    return graphs


def test_budget_sweep_shape_and_monotone_gap():
    budgets = [(5, 3), (20, 10)]
    rows = run_budget_sweep(sample_graphs(), "sample", 3, budgets, [0, 1, 2], Params())
    assert [ (r.swarm_size, r.iterations) for r in rows ] == budgets
    # The exhaustive optimum is the same reference for every budget.
    assert rows[0].exhaustive_fitness == rows[1].exhaustive_fitness == pytest.approx(0.1833333, abs=1e-6)
    # A larger budget does not do worse (gap non-increasing).
    assert rows[-1].fitness_gap_pct <= rows[0].fitness_gap_pct + 1e-9
    for r in rows:
        assert r.evaluations == r.swarm_size * (1 + r.iterations)
        assert r.pso_fitness_mean <= r.exhaustive_fitness + 1e-9


def test_budget_csv():
    rows = run_budget_sweep(sample_graphs(), "sample", 3, [(5, 3), (10, 5)], [0, 1], Params())
    csv = budget_to_csv(rows)
    assert csv.splitlines()[0].startswith("dataset,graph_limit,swarm_size,iterations,evaluations")
