"""Multi-objective machinery: dominance, Pareto fronts, hypervolume, HVR, C-metric.

The volumes are checked against hand-computed values, since every later multi-objective
comparison rests on them being right.
"""

import pytest

from wfsmbench.core import Graph
from wfsmbench.measure import Evaluation
from wfsmbench.measure.pareto import (
    bounds_of,
    coverage,
    dominates,
    hypervolume,
    hypervolume_ratio,
    normalize,
    objectives,
    pareto_front,
)


def test_objectives_negate_size_so_all_are_maximized():
    pattern = Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)])
    evaluation = Evaluation(support=2, atw=1.2, wsup=2.4)
    assert objectives(evaluation, pattern) == (2.4, 1.2, -5.0)   # 3 nodes + 2 edges


def test_dominance():
    assert dominates((2.0, 2.0), (1.0, 1.0))
    assert dominates((2.0, 1.0), (1.0, 1.0))
    assert not dominates((1.0, 1.0), (1.0, 1.0))     # equal is not domination
    assert not dominates((2.0, 0.0), (1.0, 1.0))     # a trade-off dominates nothing


def test_pareto_front_keeps_only_non_dominated():
    points = [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (1.0, 1.0), (3.0, 1.0)]
    front = pareto_front(points)
    assert set(front) == {(3.0, 1.0), (1.0, 3.0), (2.0, 2.0)}   # (1,1) dominated, dup gone


def test_hypervolume_1d_and_2d_hand_computed():
    assert hypervolume([(3.0,)], (1.0,)) == pytest.approx(2.0)
    # Two-point staircase over reference (0,0): boxes 1x3 and 3x1 overlapping in 1x1.
    assert hypervolume([(1.0, 3.0), (3.0, 1.0)], (0.0, 0.0)) == pytest.approx(5.0)
    # A dominated point adds nothing.
    assert hypervolume([(1.0, 3.0), (3.0, 1.0), (1.0, 1.0)], (0.0, 0.0)) == pytest.approx(5.0)
    # Points at or below the reference contribute nothing.
    assert hypervolume([(0.0, 5.0)], (0.0, 0.0)) == 0.0
    assert hypervolume([], (0.0, 0.0)) == 0.0


def test_hypervolume_3d_hand_computed():
    # A single box from the reference corner: 2 * 3 * 4.
    assert hypervolume([(2.0, 3.0, 4.0)], (0.0, 0.0, 0.0)) == pytest.approx(24.0)
    # Two disjoint-in-z slabs: (2x2 over z 0..1) + (1x1 extra height) checked by slicing.
    volume = hypervolume([(2.0, 2.0, 1.0), (1.0, 1.0, 2.0)], (0.0, 0.0, 0.0))
    assert volume == pytest.approx(2.0 * 2.0 * 1.0 + 1.0 * 1.0 * 1.0)


def test_normalize_and_bounds():
    points = [(0.0, 10.0), (5.0, 20.0)]
    bounds = bounds_of(points)
    assert bounds == [(0.0, 5.0), (10.0, 20.0)]
    assert normalize(points, bounds) == [(0.0, 0.0), (1.0, 1.0)]
    # A flat objective cannot be scaled, so it maps to 1.0 rather than dividing by zero.
    assert normalize([(3.0,)], [(3.0, 3.0)]) == [(1.0,)]


def test_hypervolume_ratio_bounds():
    reference = [(1.0, 3.0), (3.0, 1.0), (2.0, 2.0)]
    assert hypervolume_ratio(reference, reference) == pytest.approx(1.0)
    assert hypervolume_ratio([], reference) == 0.0
    assert hypervolume_ratio(reference, []) == 0.0
    partial = hypervolume_ratio([(2.0, 2.0)], reference)
    assert 0.0 < partial < 1.0        # one interior point recovers part of the front
    # A front missing an extreme corner must score below the full front. This only holds
    # because the reference corner sits outside the nadir (see the margin argument).
    assert hypervolume_ratio([(1.0, 3.0), (2.0, 2.0)], reference) < 1.0


def test_hypervolume_ratio_rewards_spread_not_just_one_good_point():
    """Two points spanning the trade-off beat a single point of the same peak quality."""
    reference = [(1.0, 3.0), (3.0, 1.0), (2.0, 2.0)]
    spread = hypervolume_ratio([(1.0, 3.0), (3.0, 1.0)], reference)
    single = hypervolume_ratio([(1.0, 3.0)], reference)
    assert spread > single


def test_coverage_c_metric_is_asymmetric():
    better = [(3.0, 3.0)]
    worse = [(1.0, 1.0), (2.0, 2.0)]
    assert coverage(better, worse) == pytest.approx(1.0)   # A covers all of B
    assert coverage(worse, better) == pytest.approx(0.0)   # B covers none of A
    assert coverage(worse, worse) == pytest.approx(1.0)    # weak dominance includes equality
    assert coverage([], worse) == pytest.approx(0.0)
    assert coverage(better, []) == 0.0


def test_pareto_comparison_on_the_sample():
    """The exhaustive front is the reference, so it must score HVR 1.0 and cover itself,
    and no method may exceed it."""
    from pathlib import Path
    from wfsmbench.config import Params
    from wfsmbench.experiments import pareto_to_csv, run_pareto_comparison
    from wfsmbench.io import load_dataset

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
    graphs, _ = load_dataset(sample)
    rows = run_pareto_comparison(graphs, "sample", 3, [0, 1], Params())
    by_method = {r.method: r for r in rows}

    assert "exhaustive" in by_method
    assert by_method["exhaustive"].hvr_mean == pytest.approx(1.0)
    assert by_method["exhaustive"].coverage_of_reference == pytest.approx(1.0)
    for row in rows:
        assert 0.0 <= row.hvr_mean <= 1.0 + 1e-9
        assert 0.0 <= row.coverage_of_reference <= 1.0 + 1e-9
        assert row.front_size >= 0

    csv = pareto_to_csv(rows)
    assert csv.startswith("dataset,graph_limit,max_size,method")
    assert len(csv.strip().splitlines()) == len(rows) + 1
