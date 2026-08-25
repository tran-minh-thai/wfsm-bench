"""Method-vs-pattern-size sweep, locked on the sample."""

from pathlib import Path

from wfsmbench.config import Params
from wfsmbench.experiments import run_size_sweep
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def sample_graphs():
    graphs, _ = load_dataset(SAMPLE)
    return graphs


def test_size_sweep_tags_each_size_and_method():
    rows = run_size_sweep(sample_graphs(), "sample", 3, [2, 3], [0, 1], Params(),
                          exhaustive_max_size=3)
    sizes = {r.max_size for r in rows}
    assert sizes == {2, 3}
    for size in (2, 3):
        methods = {r.method for r in rows if r.max_size == size}
        assert {"exhaustive", "greedy", "pso", "random", "genetic"} <= methods
    # Nothing beats the exhaustive optimum at each size.
    for size in (2, 3):
        cells = {r.method: r for r in rows if r.max_size == size}
        optimum = cells["exhaustive"].fitness_mean
        for r in cells.values():
            assert r.fitness_mean <= optimum + 1e-9


