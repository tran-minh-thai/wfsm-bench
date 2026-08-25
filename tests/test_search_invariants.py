"""Invariants every search must satisfy, checked for all of them at once.

These exist because the unit tests did not catch the bugs that mattered. A search can pass
every correctness test -- returns frequent patterns, never beats the optimum, reproducible
per seed -- while doing almost no searching: velocities that move nothing, budgets that
different methods count differently, an initial population that ignores the seed. Each check
below corresponds to a defect that was actually found this way:

- the velocity operator dropped its moves silently, so the swarm drifted at random;
- the colony charged its budget only for new patterns while the swarm re-paid for repeats,
  giving one of them nearly twice the search for the same nominal allowance;
- the swarm always started from the same templates, so runs were correlated;
- the genetic baseline carried half its population unchanged and converged after a third of
  its budget, understating the baseline everything else was measured against.

They are deliberately loose: the point is to catch a method that is not really searching,
not to pin down its performance.
"""

from pathlib import Path

import pytest

import wfsmbench.pso.optimizer as optimizer_module
from wfsmbench.baselines import (mine_aco, mine_genetic, mine_local_search,
                                 mine_random)
from wfsmbench.config import Params
from wfsmbench.core.signature import canonical_signature
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics
from wfsmbench.pso import optimize, optimize_clpso, optimize_multiswarm

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
BUDGET = 60
PARAMS = Params(swarm_size=6, iterations=9, min_support=2, max_nodes=4, max_edges=4)

# Every search under one calling convention: run(graphs, stats, params, seed, budget).
SEARCHES = [
    ("pso", lambda g, s, p, seed, b: optimize(g, s, p, seed, max_evaluations=b)),
    ("clpso", lambda g, s, p, seed, b: optimize_clpso(g, s, p, seed, max_evaluations=b)),
    ("multiswarm",
     lambda g, s, p, seed, b: optimize_multiswarm(g, s, p, seed, max_evaluations=b)),
    ("aco", lambda g, s, p, seed, b: mine_aco(g, s, p, seed, max_evaluations=b)),
    ("genetic", lambda g, s, p, seed, b: mine_genetic(g, s, p, seed, max_evaluations=b)),
    ("random", lambda g, s, p, seed, b: mine_random(g, s, p, seed, b)),
    ("local_search",
     lambda g, s, p, seed, b: mine_local_search(g, s, p, seed, max_evaluations=b)),
]


def setup():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def scored_signatures(run, graphs, stats, params, seed, budget=BUDGET) -> set[str]:
    """Which patterns a run actually scores -- its trajectory, not just what it reports."""
    seen: set[str] = set()
    real = optimizer_module.evaluate

    def spy(particle, hosts, st, pr):
        seen.add(canonical_signature(particle.pattern))
        return real(particle, hosts, st, pr)

    optimizer_module.evaluate = spy
    try:
        run(graphs, stats, params, seed, budget)
    finally:
        optimizer_module.evaluate = real
    return seen


def distinct_scored(run, graphs, stats, params, seed, budget=BUDGET) -> int:
    """How many distinct patterns a run scores -- its real unit of work."""
    return len(scored_signatures(run, graphs, stats, params, seed, budget))


@pytest.mark.parametrize("name,run", SEARCHES)
def test_a_search_explores_a_meaningful_share_of_its_budget(name, run):
    """The sample's space is small enough to exhaust, so a method should either spend most of
    its allowance or stop because there is genuinely nothing left -- never a small fraction
    of it while patterns remain unseen."""
    graphs, stats = setup()
    counts = [distinct_scored(run, graphs, stats, PARAMS, seed) for seed in range(3)]
    assert min(counts) >= 5, f"{name} scored almost nothing: {counts}"


# Seed-dependence has to be checked on a budget too small to exhaust the sample's space.
# Given enough allowance every method finds the same complete answer, which is correct
# behaviour and would hide a search whose randomness never reaches it.
# One starting pattern out of several candidates: the pick itself has to vary. A wider
# swarm would score every candidate whatever the order, hiding the very thing being checked.
TIGHT = Params(swarm_size=1, iterations=0, min_support=2, max_nodes=4, max_edges=4)
TIGHT_BUDGET = 1



@pytest.mark.parametrize("name,run", SEARCHES)
def test_a_search_depends_on_its_seed(name, run):
    """Different seeds must take different routes. This looks at what a run scores rather
    than what it reports: on a space small enough to exhaust every method ends up with the
    same answer, which is correct and would hide randomness that never reaches the search.
    """
    graphs, stats = setup()
    routes = {frozenset(scored_signatures(run, graphs, stats, TIGHT, seed, TIGHT_BUDGET))
              for seed in range(12)}
    assert len(routes) > 1, f"{name} ignores its seed"


@pytest.mark.parametrize("name,run", SEARCHES)
def test_a_search_is_reproducible_for_one_seed(name, run):
    graphs, stats = setup()
    first = run(graphs, stats, PARAMS, 3, BUDGET)
    second = run(graphs, stats, PARAMS, 3, BUDGET)
    assert [c.signature for c in first.top_k] == [c.signature for c in second.top_k]


@pytest.mark.parametrize("name,run", SEARCHES)
def test_a_bigger_budget_never_hurts(name, run):
    """More allowance cannot produce a worse best pattern: the archive only ever keeps its
    best, so a method that regresses is discarding work."""
    graphs, stats = setup()
    lean = run(graphs, stats, PARAMS, 0, 8)
    rich = run(graphs, stats, PARAMS, 0, BUDGET)
    if lean.gbest and rich.gbest:
        assert rich.gbest.fitness >= lean.gbest.fitness - 1e-12


@pytest.mark.parametrize("name,run", SEARCHES)
def test_a_search_terminates_on_an_unreachable_budget(name, run):
    """The sample exhausts long before a huge budget, so every method must notice and stop
    rather than spin on patterns it has already scored."""
    graphs, stats = setup()
    result = run(graphs, stats, PARAMS, 0, 10_000)
    assert result.gbest is not None
