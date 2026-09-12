"""All seven searched methods maximise the same scalarised objective, on the same anchors.

The manuscript says so in the paragraph that joins its two halves: the comparison protocol is
only fair if the methods are optimising one function, and the analysis that predicts where the
maximum sits assumes the two normalisation anchors are constants of the study rather than
per-method settings. A reviewer asked which quantity the methods maximise, and the honest
answer had to come from the code rather than from the claim.

Reading the code answers it once. This asks it of every run instead: each method is given the
SAME Params, and the fitness it reports for the pattern it returns is recomputed here, from
the shared objective, and must match to the bit. A method scoring on a variant -- a different
penalty, an anchor recomputed from its own sample, a quantum of its own -- comes out as a
different number and fails.

Cheap on purpose: the sample dataset and a budget of a few dozen evaluations. What is under
test is which function is being climbed, not how well anyone climbs it.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wfsmbench.baselines import mine_aco, mine_genetic, mine_local_search, mine_random
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params
from wfsmbench.io import load_dataset
from wfsmbench.measure import fitness as score_fitness
from wfsmbench.preprocess import build_statistics
from wfsmbench.pso import optimize, optimize_clpso, optimize_multiswarm

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
BUDGET = 40

# The seven the manuscript races, dispatched exactly as scripts/run_weight_mode.py does.
# mine_random takes its budget positionally; the rest take it by keyword. That difference is
# in the runner too, and copying it here keeps this test running what the study runs.
METHODS = {
    "pso": lambda d, s, p, seed: optimize(d, s, p, seed, max_evaluations=BUDGET),
    "clpso": lambda d, s, p, seed: optimize_clpso(d, s, p, seed, max_evaluations=BUDGET),
    "multiswarm": lambda d, s, p, seed: optimize_multiswarm(d, s, p, seed,
                                                            max_evaluations=BUDGET),
    "aco": lambda d, s, p, seed: mine_aco(d, s, p, seed, max_evaluations=BUDGET),
    "genetic": lambda d, s, p, seed: mine_genetic(d, s, p, seed, max_evaluations=BUDGET),
    "local_search": lambda d, s, p, seed: mine_local_search(d, s, p, seed,
                                                            max_evaluations=BUDGET),
    "random": lambda d, s, p, seed: mine_random(d, s, p, seed, BUDGET),
}


def _du_lieu():
    graphs, _ = load_dataset(SAMPLE)
    return graphs, build_statistics(graphs)


def _tham_so() -> Params:
    """One Params for every method, built the way the study builds it."""
    return Params(max_nodes=4, max_edges=4, min_support=2,
                  size_reference=STUDY_SIZE_REFERENCE, swarm_size=5, iterations=3)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_the_reported_fitness_is_the_shared_objective(method: str) -> None:
    graphs, stats = _du_lieu()
    params = _tham_so()
    result = METHODS[method](graphs, stats, params, 0)
    best = result.gbest
    assert best is not None, (
        f"{method} returned no pattern at a budget of {BUDGET}; this check cannot say what it "
        f"was maximising")
    pattern = best.particle.pattern
    lai = score_fitness(best.evaluation, pattern.num_nodes(), pattern.num_edges(),
                        stats, params)
    assert lai == pytest.approx(best.fitness, abs=1e-12), (
        f"{method} reports {best.fitness!r} for the pattern it returned, and the shared "
        f"objective scores that same pattern {lai!r}. The method is climbing a different "
        f"function from the one the analysis is about.")


def test_no_method_alters_the_parameters_it_was_given() -> None:
    """The anchors are constants of the study, which is the half of the claim a value check
    cannot see: a method that quietly rescaled its own copy would still report a matching
    number for its own pattern."""
    graphs, stats = _du_lieu()
    params = _tham_so()
    truoc = replace(params)
    for method, chay in sorted(METHODS.items()):
        chay(graphs, stats, params, 0)
        assert params == truoc, f"{method} mutated the shared parameters"


def test_the_anchors_are_the_same_for_every_method() -> None:
    """Same pattern, same anchors, whichever method found it.

    Scored through one Params, so the normalisation is by construction shared -- what this
    rules out is a method returning a Candidate whose fitness was computed against anchors
    taken from its own sample of the data, which would show up as two different values for
    one pattern.
    """
    graphs, stats = _du_lieu()
    params = _tham_so()
    diem: dict[str, float] = {}
    for method, chay in sorted(METHODS.items()):
        best = chay(graphs, stats, params, 0).gbest
        if best is None:
            continue
        pattern = best.particle.pattern
        khoa = pattern.canonical_key() if hasattr(pattern, "canonical_key") else str(pattern)
        lai = score_fitness(best.evaluation, pattern.num_nodes(), pattern.num_edges(),
                            stats, params)
        cu = diem.setdefault(khoa, lai)
        assert cu == pytest.approx(lai, abs=1e-12), (
            f"one pattern scores {cu} for one method and {lai} for {method}; the two are not "
            f"normalising against the same anchors")
    assert diem, "no method returned a pattern; the check compared nothing"
