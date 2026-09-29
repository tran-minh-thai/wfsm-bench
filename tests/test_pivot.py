"""Pivot scoring and selection, locked on the sample.

Label encoding: A=0, B=1, C=2; edge label 0. Default coefficients (D3): lambda=0.25 each.
"""

from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.core import Graph
from wfsmbench.io import load_dataset
from wfsmbench.pivot import pivot_score, select_pivot
from wfsmbench.preprocess import EdgeTemplate, build_statistics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"


def sample_stats():
    graphs, _ = load_dataset(SAMPLE)
    return build_statistics(graphs)


def test_template_mean_weight():
    stats = sample_stats()
    assert stats.template_weight(EdgeTemplate.of(0, 1, 0)) == pytest.approx(0.8)   # A-B: (0.8+0.6+1.0)/3
    assert stats.template_weight(EdgeTemplate.of(1, 2, 0)) == pytest.approx(0.5)   # B-C: (0.4+0.6)/2
    assert stats.template_weight(EdgeTemplate.of(0, 2, 0)) == 0.0                  # no A-C edge


def test_pivot_scores_hand_computed():
    stats = sample_stats()
    params = Params()
    path = Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)])  # A(0)-B(1)-C(2)
    # A: 0.25*(deg1 + wdeg0.8 + cent0.5 + lab1.0) = 0.825
    assert pivot_score(path, 0, stats, params) == pytest.approx(0.825)
    # B: 0.25*(deg2 + wdeg1.3 + cent1.0 + lab1.0) = 1.325
    assert pivot_score(path, 1, stats, params) == pytest.approx(1.325)
    # C: 0.25*(deg1 + wdeg0.5 + cent0.5 + lab0.6667) = 0.666667
    assert pivot_score(path, 2, stats, params) == pytest.approx(0.6666667, abs=1e-6)


def test_select_pivot_picks_central_node():
    stats = sample_stats()
    path = Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)])
    assert select_pivot(path, stats, Params()) == 1  # B, the central high-degree node


def test_pivot_tie_breaks_on_smaller_id():
    stats = sample_stats()
    # A-A edge: both endpoints have identical scores, so the smaller id wins.
    symmetric = Graph.of({0: 0, 1: 0}, [(0, 1, 0)])
    assert select_pivot(symmetric, stats, Params()) == 0


def test_empty_pattern_has_no_pivot():
    with pytest.raises(ValueError):
        select_pivot(Graph.of({}, []), sample_stats(), Params())
