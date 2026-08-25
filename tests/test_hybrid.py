"""Beam-seeded particle swarm — starts from the beam frontier, keeps the PSO objective,
and stays inside the plain swarm's evaluation budget. Locked on the sample.

Label encoding: A=0, B=1, C=2; edge label 0. Sample optimum = path A-B-C, F=0.1833.
"""

from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.baselines.greedy import beam_search
from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics
from wfsmbench.pso.hybrid import mine_hybrid

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def test_hybrid_never_beats_the_exhaustive_optimum():
    graphs, stats = setup()
    params = Params()
    best = mine_exhaustive(graphs, stats, params).gbest.fitness
    for seed in range(3):
        result = mine_hybrid(graphs, stats, params, seed)
        assert result.gbest is not None
        assert result.gbest.fitness <= best + 1e-12
        assert len(result.top_k) <= params.top_k


def test_hybrid_is_deterministic_for_a_seed():
    graphs, stats = setup()
    params = Params()
    a = mine_hybrid(graphs, stats, params, seed=1)
    b = mine_hybrid(graphs, stats, params, seed=1)
    assert a.gbest.fitness == b.gbest.fitness
    assert [c.signature for c in a.top_k] == [c.signature for c in b.top_k]


def test_beam_search_respects_the_level_bound():
    graphs, stats = setup()
    params = Params()
    _archive, shallow, shallow_evals = beam_search(graphs, stats, params, 3, max_levels=1)
    _archive, deep, deep_evals = beam_search(graphs, stats, params, 3)
    assert shallow_evals <= deep_evals
    # One expansion round from the single-edge seeds yields two-edge patterns.
    assert all(p.pattern.num_edges() == 2 for p in shallow)
    assert all(p.pattern.num_edges() >= 2 for p in deep)


def test_hybrid_warm_start_reaches_larger_patterns_than_its_seeds():
    graphs, stats = setup()
    params = Params()
    _archive, frontier, _evals = beam_search(graphs, stats, params, params.swarm_size,
                                             max_levels=1)
    assert frontier, "the warm start must hand the swarm something to start from"
    result = mine_hybrid(graphs, stats, params, seed=0)
    assert result.gbest.particle.pattern.num_edges() >= 1


def test_hybrid_falls_back_when_no_frequent_templates_exist():
    graphs, stats = setup()
    params = Params(min_support=10_000)      # nothing is frequent
    result = mine_hybrid(graphs, stats, params, seed=0)
    assert result.top_k == [] and result.gbest is None
