"""Dynamic multi-swarm PSO: sub-swarm partitioning, and the same guarantees as the basic
swarm (never beats the exhaustive optimum, deterministic per seed).

Label encoding: A=0, B=1, C=2; edge label 0. Sample optimum = path A-B-C, F=0.1833.
"""

from pathlib import Path

import numpy as np
import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics
from wfsmbench.pso.multiswarm import optimize_multiswarm, partition

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def test_partition_covers_every_particle_exactly_once():
    rng = np.random.default_rng(0)
    groups = partition(20, 4, rng)
    assert len(groups) == 4
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(20))                      # a partition, not a sample
    assert all(len(g) == 5 for g in groups)
    # More groups than particles collapses to one particle per group.
    small = partition(3, 10, rng)
    assert sorted(i for g in small for i in g) == [0, 1, 2]
    # One group reproduces the basic star topology.
    assert len(partition(6, 1, rng)) == 1


def test_multiswarm_never_beats_the_exhaustive_optimum():
    graphs, stats = setup()
    params = Params(num_subswarms=2)
    best = mine_exhaustive(graphs, stats, params).gbest.fitness
    for seed in range(4):
        result = optimize_multiswarm(graphs, stats, params, seed)
        assert result.gbest is not None
        assert result.gbest.fitness <= best + 1e-12
        assert len(result.top_k) <= params.top_k


def test_multiswarm_is_deterministic_for_a_seed():
    graphs, stats = setup()
    params = Params(num_subswarms=2)
    a = optimize_multiswarm(graphs, stats, params, seed=3)
    b = optimize_multiswarm(graphs, stats, params, seed=3)
    assert a.gbest.fitness == b.gbest.fitness
    assert [c.signature for c in a.top_k] == [c.signature for c in b.top_k]


def test_multiswarm_reports_only_frequent_patterns():
    graphs, stats = setup()
    params = Params(num_subswarms=2)
    result = optimize_multiswarm(graphs, stats, params, seed=0)
    for candidate in result.top_k:
        assert candidate.evaluation.support >= params.min_support


def test_multiswarm_params_are_validated():
    Params(num_subswarms=1, regroup_period=1)
    with pytest.raises(ValueError):
        Params(num_subswarms=0)
    with pytest.raises(ValueError):
        Params(regroup_period=0)


def test_multiswarm_falls_back_when_no_frequent_templates_exist():
    graphs, stats = setup()
    result = optimize_multiswarm(graphs, stats, Params(min_support=10_000), seed=0)
    assert result.top_k == [] and result.gbest is None
