"""P2 - dataset statistics and frequent edge templates, locked on the sample.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

from pathlib import Path

import pytest

from wfsmbench.io import load_dataset
from wfsmbench.preprocess import EdgeTemplate, build_statistics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
BZR_JSON = Path(__file__).resolve().parents[1] / "data" / "BZR_MD.json"


def sample_stats():
    graphs, _ = load_dataset(SAMPLE)
    return build_statistics(graphs)


def test_template_canonicalizes_orientation():
    assert EdgeTemplate.of(2, 1, 0) == EdgeTemplate.of(1, 2, 0)
    assert EdgeTemplate.of(1, 2, 0) == EdgeTemplate(1, 2, 0)


def test_label_frequency():
    stats = sample_stats()
    assert stats.num_graphs == 3
    assert stats.node_label_graph_freq == {0: 3, 1: 3, 2: 2}
    assert stats.label_frequency(0) == 1.0          # A in every graph
    assert stats.label_frequency(1) == 1.0          # B in every graph
    assert stats.label_frequency(2) == pytest.approx(2 / 3)  # C in two of three


def test_edge_template_support_and_frequency():
    stats = sample_stats()
    ab = EdgeTemplate.of(0, 1, 0)
    bc = EdgeTemplate.of(1, 2, 0)
    assert stats.edge_template_support[ab] == 3
    assert stats.edge_template_support[bc] == 2
    assert stats.frequent_edge_templates(min_support=2) == [ab, bc]
    assert stats.frequent_edge_templates(min_support=3) == [ab]


def test_weight_stats_and_norm_references():
    stats = sample_stats()
    assert stats.max_edge_weight == 1.0     # graph 2's A-B edge
    assert stats.min_edge_weight == 0.4     # graph 0's B-C edge
    # D2 fixed normalization anchors, scaled by the study constant Params.size_reference
    # (never by the search bound max_edges -- see config.py).
    assert stats.atw_reference(size_reference=6) == 6.0
    assert stats.wsup_reference(size_reference=6) == 18.0        # 3 graphs * 6.0
    assert stats.wsup_reference(size_reference=6, num_graphs_ref=10) == 60.0


@pytest.mark.skipif(not BZR_JSON.exists(), reason="BZR_MD.json not generated")
def test_bzr_md_statistics():
    graphs, _ = load_dataset(BZR_JSON)
    stats = build_statistics(graphs)
    assert stats.num_graphs == 306
    assert stats.max_edge_weight == pytest.approx(5.0)
    assert stats.min_edge_weight == pytest.approx(0.5)
    assert len(stats.frequent_edge_templates(min_support=1)) > 0
