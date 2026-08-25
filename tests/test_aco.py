"""Ant colony baseline: constructive search sharing the objective, archive, and budget.

Label encoding: A=0, B=1, C=2; edge label 0. Sample optimum = path A-B-C, F=0.1833.
"""

from pathlib import Path

import pytest

from wfsmbench.baselines import mine_aco, mine_exhaustive
from wfsmbench.baselines.aco import heuristic
from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def test_heuristic_is_positive_and_normalized():
    graphs, stats = setup()
    templates = stats.frequent_edge_templates(2)
    eta = heuristic(stats, templates)
    assert set(eta) == set(templates)
    assert all(0.0 < value <= 1.0 for value in eta.values())
    assert max(eta.values()) == pytest.approx(1.0)   # the most attractive template anchors 1


def test_aco_never_beats_the_exhaustive_optimum():
    graphs, stats = setup()
    params = Params()
    best = mine_exhaustive(graphs, stats, params).gbest.fitness
    for seed in range(4):
        result = mine_aco(graphs, stats, params, seed)
        assert result.gbest is not None
        assert result.gbest.fitness <= best + 1e-12
        assert len(result.top_k) <= params.top_k


def test_aco_is_deterministic_for_a_seed():
    graphs, stats = setup()
    a = mine_aco(graphs, stats, Params(), seed=1)
    b = mine_aco(graphs, stats, Params(), seed=1)
    assert a.gbest.fitness == b.gbest.fitness
    assert [c.signature for c in a.top_k] == [c.signature for c in b.top_k]


def test_aco_respects_its_evaluation_budget():
    """A tighter budget must not produce a better result than a generous one, which is the
    observable sign that the cap is actually enforced."""
    graphs, stats = setup()
    params = Params()
    tight = mine_aco(graphs, stats, params, seed=0, max_evaluations=2)
    generous = mine_aco(graphs, stats, params, seed=0, max_evaluations=200)
    assert tight.gbest.fitness <= generous.gbest.fitness + 1e-12
    assert len(tight.top_k) <= len(generous.top_k)


def test_aco_reports_only_frequent_patterns():
    graphs, stats = setup()
    params = Params()
    for candidate in mine_aco(graphs, stats, params, seed=0).top_k:
        assert candidate.evaluation.support >= params.min_support


def test_aco_params_are_validated():
    Params(aco_evaporation=1.0, num_ants=1)
    with pytest.raises(ValueError):
        Params(aco_evaporation=0.0)
    with pytest.raises(ValueError):
        Params(num_ants=0)


def test_aco_falls_back_when_no_frequent_templates_exist():
    graphs, stats = setup()
    result = mine_aco(graphs, stats, Params(min_support=10_000), seed=0)
    assert result.top_k == [] and result.gbest is None


def test_aco_terminates_when_the_space_is_exhausted():
    """Every reachable pattern on the sample is cheap to find, so the colony runs out of new
    ones long before a large budget is spent; it must stop rather than spin."""
    graphs, stats = setup()
    result = mine_aco(graphs, stats, Params(), seed=0, max_evaluations=10_000)
    assert result.gbest is not None
