"""P4 - subgraph matching (VF2 monomorphism). Closes the loop: real matcher output
fed to the P3 measure must reproduce the sample ground truth.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

import json
from pathlib import Path

import pytest

from wfsmbench.core import Graph
from wfsmbench.io import load_dataset
from wfsmbench.match import embedding_weights, match_database
from wfsmbench.measure import Evaluation

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
GROUND_TRUTH = Path(__file__).resolve().parents[1] / "data" / "sample" / "ground_truth.json"


def sample_graphs():
    graphs, _ = load_dataset(SAMPLE)
    return graphs


def edge_pattern(la, lb, edge_label=0):
    return Graph.of({0: la, 1: lb}, [(0, 1, edge_label)])


def ground_truth():
    return {p["id"]: p for p in json.loads(GROUND_TRUTH.read_text())["patterns"]}


def test_matcher_reproduces_ground_truth():
    truth = ground_truth()
    graphs = sample_graphs()

    p1 = edge_pattern(0, 1)                                    # A-B
    ev1 = Evaluation.from_embeddings(match_database(p1, graphs))
    assert ev1.support == truth["P1"]["support"]
    assert ev1.wsup == pytest.approx(truth["P1"]["wsup"])

    p2 = Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)])  # path A-B-C
    ev2 = Evaluation.from_embeddings(match_database(p2, graphs))
    assert ev2.support == truth["P2"]["support"]
    assert ev2.wsup == pytest.approx(truth["P2"]["wsup"])


def test_absent_pattern_has_no_support():
    graphs = sample_graphs()
    assert match_database(edge_pattern(0, 2), graphs) == {}   # no A-C edge anywhere


def test_monomorphism_allows_extra_host_edges():
    # Complete triangle: matching edge A-B must succeed despite the extra B-C and A-C edges.
    triangle = Graph.of({0: 0, 1: 1, 2: 2},
                        [(0, 1, 0, 0.5), (1, 2, 0, 0.3), (0, 2, 0, 0.9)], graph_id=0)
    assert embedding_weights(edge_pattern(0, 1), triangle) == [[0.5]]


def test_edge_label_constrains_match():
    host = Graph.of({0: 0, 1: 1}, [(0, 1, 1, 0.5)], graph_id=0)  # A-B with edge label 1
    assert embedding_weights(edge_pattern(0, 1, edge_label=0), host) == []
    assert embedding_weights(edge_pattern(0, 1, edge_label=1), host) == [[0.5]]


def test_automorphic_mappings_collapse_to_one_embedding():
    # Symmetric pattern A-A over a single A-A host edge is one occurrence, not two.
    pattern = Graph.of({0: 0, 1: 0}, [(0, 1, 0)])
    host = Graph.of({0: 0, 1: 0}, [(0, 1, 0, 0.7)], graph_id=0)
    assert embedding_weights(pattern, host) == [[0.7]]
