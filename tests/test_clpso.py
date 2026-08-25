"""Comprehensive-Learning PSO variant — shares the objective, archive, and validity of the
basic swarm, never beats the exhaustive optimum, and is deterministic per seed. Sample.

Label encoding: A=0, B=1, C=2; edge label 0. Sample optimum = path A-B-C, F=0.1833.
"""

from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics
from wfsmbench.pso.clpso import learning_probabilities, optimize_clpso

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def test_learning_probabilities_span_the_clpso_range():
    pc = learning_probabilities(20)
    assert len(pc) == 20
    assert pc[0] == pytest.approx(0.05, abs=1e-9)
    assert pc[-1] == pytest.approx(0.5, abs=1e-9)
    assert all(a <= b for a, b in zip(pc, pc[1:]))     # monotonically increasing
    assert learning_probabilities(1) == [0.0]


def test_clpso_never_beats_the_exhaustive_optimum():
    graphs, stats = setup()
    params = Params()
    best = mine_exhaustive(graphs, stats, params).gbest.fitness
    for seed in range(4):
        result = optimize_clpso(graphs, stats, params, seed)
        assert result.gbest is not None
        assert result.gbest.fitness <= best + 1e-12
        assert len(result.top_k) <= params.top_k


def test_clpso_is_deterministic_for_a_seed():
    graphs, stats = setup()
    params = Params()
    a = optimize_clpso(graphs, stats, params, seed=2)
    b = optimize_clpso(graphs, stats, params, seed=2)
    assert a.gbest.fitness == b.gbest.fitness
    assert [c.signature for c in a.top_k] == [c.signature for c in b.top_k]


def test_clpso_reports_only_frequent_valid_patterns():
    graphs, stats = setup()
    params = Params()
    result = optimize_clpso(graphs, stats, params, seed=0)
    for candidate in result.top_k:
        assert candidate.evaluation.support >= params.min_support
        assert candidate.particle.pattern.num_edges() >= 1


def test_clpso_falls_back_when_no_frequent_templates_exist():
    graphs, stats = setup()
    result = optimize_clpso(graphs, stats, Params(min_support=10_000), seed=0)
    assert result.top_k == [] and result.gbest is None


def test_the_refresh_gap_only_means_anything_against_the_round_count():
    """Holding the exemplar for more rounds than a run has pins it for the whole run. The
    published gap of 7 assumes thousands of rounds; on a handful it throttles the search."""
    from dataclasses import replace

    graphs, stats = setup()
    params = Params(swarm_size=4, iterations=2, min_support=2, max_nodes=3, max_edges=3)
    short_run_rounds = 1 + params.iterations
    assert params.clpso_refresh_gap < short_run_rounds, \
        "the default gap must leave room to rebuild the exemplar at least once"

    # Both settings must still run and respect the optimum; only their quality differs.
    for gap in (1, 7):
        result = optimize_clpso(graphs, stats, replace(params, clpso_refresh_gap=gap), seed=0)
        assert result.gbest is not None


def test_clpso_refresh_gap_is_validated():
    Params(clpso_refresh_gap=1), Params(clpso_refresh_gap=50)
    with pytest.raises(ValueError):
        Params(clpso_refresh_gap=0)
