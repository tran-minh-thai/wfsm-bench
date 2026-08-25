"""P8 - experiment runner: aggregation, determinism, CSV, and all pivot strategies.

Runs on the tiny sample so the tests stay fast; runtime values are not asserted.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.experiments import Aggregate, run_experiment, to_csv
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def sample_graphs():
    graphs, _ = load_dataset(SAMPLE)
    return graphs


def test_run_experiment_aggregates_over_seeds():
    graphs = sample_graphs()
    agg = run_experiment(graphs, "sample", graph_limit=3, seeds=[0, 1, 2], params=Params())
    assert isinstance(agg, Aggregate)
    assert agg.dataset == "sample"
    assert agg.graph_limit == 3
    assert agg.num_seeds == 3
    assert agg.pivot_strategy == "scientific"
    assert agg.fitness_mean >= 0.17 - 1e-9      # at least the A-B seed
    assert agg.support_mean >= 2
    assert agg.mode_nodes >= 2 and agg.mode_edges >= 1


def test_run_experiment_is_deterministic():
    graphs = sample_graphs()
    a = run_experiment(graphs, "sample", 3, [0, 1, 2], Params())
    b = run_experiment(graphs, "sample", 3, [0, 1, 2], Params())
    assert a.fitness_mean == b.fitness_mean
    assert a.wsup_mean == b.wsup_mean
    assert (a.mode_nodes, a.mode_edges) == (b.mode_nodes, b.mode_edges)


def test_all_pivot_strategies_run():
    graphs = sample_graphs()
    for strategy in Params._PIVOT_STRATEGIES:
        agg = run_experiment(graphs, "sample", 3, [0, 1], replace(Params(), pivot_strategy=strategy))
        assert agg.pivot_strategy == strategy
        assert agg.fitness_mean == agg.fitness_mean  # not NaN


def test_to_csv_shape():
    graphs = sample_graphs()
    rows = [run_experiment(graphs, "sample", 3, [0, 1], Params())]
    csv = to_csv(rows)
    lines = csv.strip().split("\n")
    assert lines[0].startswith("dataset,graph_limit,pivot_strategy,num_seeds")
    assert len(lines) == 2                         # header + one row
    assert lines[1].startswith("sample,3,scientific,2,")


def test_invalid_pivot_strategy_rejected():
    with pytest.raises(ValueError):
        Params(pivot_strategy="bogus")
