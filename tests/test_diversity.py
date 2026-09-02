"""Solution-diversity metric and the fitness/diversity view.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.core import Graph
from wfsmbench.experiments import run_method_comparison
from wfsmbench.io import load_dataset
from wfsmbench.measure import distinct_sizes, mean_pairwise_diversity
from wfsmbench.measure.diversity import select_diverse

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def test_mean_pairwise_diversity():
    ab = Graph.of({0: 0, 1: 1}, [(0, 1, 0)])
    bc = Graph.of({0: 1, 1: 2}, [(0, 1, 0)])
    path = Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)])
    # d(AB,BC)=1, d(AB,path)=0.5, d(BC,path)=0.5 -> mean 2/3.
    assert mean_pairwise_diversity([ab, bc, path]) == pytest.approx(2 / 3)
    assert mean_pairwise_diversity([ab]) == 0.0                    # single pattern
    assert mean_pairwise_diversity([]) == 0.0
    # Same template content -> distance 0.
    ab_relabelled = Graph.of({5: 0, 7: 1}, [(5, 7, 0)])
    assert mean_pairwise_diversity([ab, ab_relabelled]) == 0.0
    assert distinct_sizes([ab, bc, path]) == 2                     # (2,1) and (3,2)


def test_select_diverse_weight_zero_is_plain_top_k():
    ab = Graph.of({0: 0, 1: 1}, [(0, 1, 0)])
    ab2 = Graph.of({3: 0, 4: 1}, [(3, 4, 0)])     # same templates as ab
    bc = Graph.of({0: 1, 1: 2}, [(0, 1, 0)])      # disjoint templates
    scored = [(0.9, ab), (0.8, ab2), (0.3, bc)]
    assert select_diverse(scored, 2, weight=0.0) == [0, 1]
    # With a diversity pull the near-duplicate loses to the structurally novel pattern.
    assert select_diverse(scored, 2, weight=0.7) == [0, 2]
    # The fittest candidate is always picked first, whatever the weight.
    assert select_diverse(scored, 3, weight=1.0)[0] == 0
    assert sorted(select_diverse(scored, 9, weight=0.5)) == [0, 1, 2]  # k beyond the pool
    assert select_diverse([], 3, weight=0.5) == []


def test_diversity_weight_is_validated():
    Params(diversity_weight=0.0), Params(diversity_weight=1.0)
    with pytest.raises(ValueError):
        Params(diversity_weight=1.5)


COLLECTION = Path(__file__).resolve().parents[1] / "data" / "BZR_bond.json"


def _run(graphs, stats, **kw):
    from wfsmbench.pso.optimizer import optimize
    return optimize(graphs, stats, Params(**kw), seed=0)


def test_the_diversity_rule_leaves_the_search_alone():
    """The reporting rule changes what is reported, not what is found.

    This is the half the sample collection can establish, and it is worth having on its own:
    it is the claim that lets every earlier result stand at diversity_weight = 0.
    """
    from wfsmbench.preprocess.statistics import build_statistics
    graphs, _ = load_dataset(SAMPLE)
    stats = build_statistics(graphs)
    assert _run(graphs, stats).gbest.fitness == _run(graphs, stats, diversity_weight=0.5).gbest.fitness


def test_the_sample_collection_cannot_exercise_the_diversity_rule():
    """Why the check above stops where it does, measured rather than assumed.

    The sample yields three patterns in total, so a top-k of three has nothing to choose
    between and the diversity term cannot move anything. Recording that here stops anyone
    re-adding a diversity assertion to the sample and reading its pass as evidence.
    """
    from wfsmbench.preprocess.statistics import build_statistics
    graphs, _ = load_dataset(SAMPLE)
    stats = build_statistics(graphs)
    ket_qua = _run(graphs, stats, top_k=20)
    assert len(ket_qua.top_k) <= 3, (
        f"the sample now yields {len(ket_qua.top_k)} patterns at top_k=20; it may be able to "
        f"exercise the diversity rule after all, so move the real check back onto it")


@pytest.mark.skipif(not COLLECTION.exists(),
                    reason=f"needs a real collection at {COLLECTION.name}; the sample has "
                           f"three patterns and cannot exercise this rule")
def test_diverse_archive_raises_top_k_diversity():
    """SUA 2026-08-27. Two faults in one line, and the second is why this moved collections.

    The assertion was `d_diverse >= d_plain`, which EQUALITY satisfies -- so a diversity term
    that did nothing at all would pass a test named for raising diversity. And it ran on the
    sample, where the term provably cannot act: measured, plain and diverse both scored
    0.666667 there, at every top_k tried.

    On a real collection the term acts, and by a wide margin. The assertion is strict now, and
    both numbers are printed rather than only compared.
    """
    from wfsmbench.preprocess.statistics import build_statistics
    graphs, _ = load_dataset(COLLECTION)
    graphs = graphs[:60]
    stats = build_statistics(graphs)
    plain = _run(graphs, stats, top_k=10, swarm_size=8, iterations=6)
    diverse = _run(graphs, stats, top_k=10, swarm_size=8, iterations=6, diversity_weight=0.5)

    assert len(plain.top_k) >= 5, (
        f"only {len(plain.top_k)} pattern(s) reported; too few to choose between, which is "
        f"the condition that made the previous version unable to fail")
    assert plain.gbest.fitness == diverse.gbest.fitness, "the reporting rule changed the search"

    d_plain = mean_pairwise_diversity([c.particle.pattern for c in plain.top_k])
    d_diverse = mean_pairwise_diversity([c.particle.pattern for c in diverse.top_k])
    print(f"\n[diversity] plain {d_plain:.6f} -> diverse {d_diverse:.6f} "
          f"over {len(plain.top_k)} reported patterns")
    assert d_diverse > d_plain, (
        f"the diversity term did not raise diversity: {d_plain:.6f} -> {d_diverse:.6f}. "
        f"Equality is not enough here -- a term that does nothing produces equality.")


def test_method_comparison_reports_diversity():
    graphs, _ = load_dataset(SAMPLE)
    rows = run_method_comparison(graphs, "sample", 3, [0, 1], Params())
    for r in rows:
        assert 0.0 <= r.diversity_mean <= 1.0


