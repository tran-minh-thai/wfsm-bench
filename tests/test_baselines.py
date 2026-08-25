"""Random, greedy, and genetic baselines — each shares the PSO objective and never
beats the exhaustive optimum. Locked on the sample.

Label encoding: A=0, B=1, C=2; edge label 0. Sample optimum = path A-B-C, F=0.1833.
"""

from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive, mine_genetic, mine_greedy, mine_random
from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
PATH_SIG = "V[0,1,2]E[0-1:0;1-2:0]"


def setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def optimum(graphs, stats, params):
    return mine_exhaustive(graphs, stats, params).gbest.fitness


def test_random_search():
    graphs, stats = setup()
    params = Params()
    best = mine_random(graphs, stats, params, seed=0, max_evaluations=100)
    assert best.gbest.signature == PATH_SIG      # trivially small space -> finds the optimum
    assert best.gbest.fitness <= optimum(graphs, stats, params) + 1e-9
    # Deterministic for a given seed.
    again = mine_random(graphs, stats, params, seed=0, max_evaluations=100)
    assert best.gbest.fitness == again.gbest.fitness


def test_greedy_beam():
    graphs, stats = setup()
    params = Params()
    best = mine_greedy(graphs, stats, params, beam_width=3)
    assert best.gbest.signature == PATH_SIG      # deterministic beam finds the optimum
    assert best.gbest.fitness == pytest.approx(optimum(graphs, stats, params))
    assert [c.signature for c in best.top_k] == [c.signature for c in
            mine_greedy(graphs, stats, params, beam_width=3).top_k]


def test_genetic():
    graphs, stats = setup()
    params = Params()
    best = mine_genetic(graphs, stats, params, seed=0)
    assert best.gbest is not None
    assert best.gbest.fitness <= optimum(graphs, stats, params) + 1e-9
    assert best.gbest.fitness >= 0.17 - 1e-9      # at least the single edge A-B
    for c in best.top_k:
        assert c.evaluation.support >= params.min_support
    # Deterministic for a given seed.
    assert best.gbest.fitness == mine_genetic(graphs, stats, params, seed=0).gbest.fitness


def test_elitism_does_not_throttle_the_genetic_search():
    """Carrying half the population unchanged left so little room for new individuals that
    the search stopped at a third of its budget and well short of the optimum."""
    from dataclasses import replace

    import wfsmbench.pso.optimizer as optimizer_module
    from wfsmbench.core.signature import canonical_signature

    graphs, stats = setup()
    base = Params(swarm_size=20, iterations=8, min_support=2, max_nodes=3, max_edges=3)

    def distinct_evaluations(params):
        seen = set()
        real = optimizer_module.evaluate

        def spy(particle, hosts, st, pr):
            seen.add(canonical_signature(particle.pattern))
            return real(particle, hosts, st, pr)

        optimizer_module.evaluate = spy
        try:
            mine_genetic(graphs, stats, params, seed=0)
        finally:
            optimizer_module.evaluate = real
        return len(seen)

    lean = distinct_evaluations(replace(base, elite_fraction=0.05))
    heavy = distinct_evaluations(replace(base, elite_fraction=0.5))
    assert lean >= heavy, "lighter elitism must not explore less"


def test_elite_fraction_is_validated():
    Params(elite_fraction=0.05), Params(elite_fraction=0.9)
    with pytest.raises(ValueError):
        Params(elite_fraction=0.0)
    with pytest.raises(ValueError):
        Params(elite_fraction=1.0)
