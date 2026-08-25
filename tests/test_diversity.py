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


def test_diverse_archive_raises_top_k_diversity():
    graphs, _ = load_dataset(SAMPLE)
    from wfsmbench.pso.optimizer import optimize
    from wfsmbench.preprocess.statistics import build_statistics
    stats = build_statistics(graphs)
    plain = optimize(graphs, stats, Params(), seed=0)
    diverse = optimize(graphs, stats, Params(diversity_weight=0.5), seed=0)
    # Same search, same best pattern found; only the reported set differs.
    assert plain.gbest.fitness == diverse.gbest.fitness
    d_plain = mean_pairwise_diversity([c.particle.pattern for c in plain.top_k])
    d_diverse = mean_pairwise_diversity([c.particle.pattern for c in diverse.top_k])
    assert d_diverse >= d_plain


def test_method_comparison_reports_diversity():
    graphs, _ = load_dataset(SAMPLE)
    rows = run_method_comparison(graphs, "sample", 3, [0, 1], Params())
    for r in rows:
        assert 0.0 <= r.diversity_mean <= 1.0


