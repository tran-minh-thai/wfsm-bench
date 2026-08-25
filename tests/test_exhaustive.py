"""Exhaustive baseline: it finds the true optimum, and the PSO never beats it.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics
from wfsmbench.pso import optimize

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
PATH_SIG = "V[0,1,2]E[0-1:0;1-2:0]"
AB_SIG = "V[0,1]E[0-1:0]"


def sample_setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def test_exhaustive_finds_the_true_optimum():
    graphs, stats = sample_setup()
    result = mine_exhaustive(graphs, stats, Params())
    sigs = {c.signature for c in result.top_k}
    # Frequent patterns at min_support=2 are A-B, B-C, and the path A-B-C.
    assert len(result.top_k) == 3
    assert result.gbest.signature == PATH_SIG          # path has the highest fitness
    assert result.gbest.fitness == pytest.approx(0.1833333, abs=1e-6)
    assert AB_SIG in sigs
    for c in result.top_k:
        assert c.evaluation.support >= Params().min_support


def test_exhaustive_is_deterministic():
    graphs, stats = sample_setup()
    a = mine_exhaustive(graphs, stats, Params())
    b = mine_exhaustive(graphs, stats, Params())
    assert [(c.signature, c.fitness) for c in a.top_k] == [(c.signature, c.fitness) for c in b.top_k]


def test_pso_never_beats_the_exhaustive_optimum():
    graphs, stats = sample_setup()
    params = Params()
    exhaustive = mine_exhaustive(graphs, stats, params)
    pso = optimize(graphs, stats, params, seed=0)
    # The exhaustive optimum is an upper bound on what the PSO can find...
    assert pso.gbest.fitness <= exhaustive.gbest.fitness + 1e-9
    # ...and the PSO's best is a genuinely top-ranked pattern.
    assert pso.gbest.signature in {c.signature for c in exhaustive.top_k}
