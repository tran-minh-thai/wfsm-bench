"""Parallel execution must match serial exactly (runs are independently seeded)."""

from pathlib import Path

from wfsmbench.config import Params
from wfsmbench.experiments import run_method_comparison
from wfsmbench.experiments.parallel import resolve_workers

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def test_resolve_workers():
    assert resolve_workers(3) == 3
    assert resolve_workers(0) >= 1
    assert resolve_workers(None) >= 1


def test_parallel_matches_serial():
    from wfsmbench.io import load_dataset
    graphs, _ = load_dataset(SAMPLE)
    params = Params()
    serial = run_method_comparison(graphs, "sample", 3, [0, 1, 2], params, workers=1)
    parallel = run_method_comparison(graphs, "sample", 3, [0, 1, 2], params, workers=2)

    def summary(rows):
        return {r.method: (round(r.fitness_mean, 6), round(r.diversity_mean, 6),
                           round(r.gap_pct, 4)) for r in rows}

    assert summary(serial) == summary(parallel)
