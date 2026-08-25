"""P0 - core types and canonical signature, locked on the sample dataset.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

import json
from pathlib import Path

import pytest

from wfsmbench.core import Graph, Particle, canonical_signature
from wfsmbench.io import load_dataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
GROUND_TRUTH = Path(__file__).resolve().parents[1] / "data" / "sample" / "ground_truth.json"


def path_abc(ids=(0, 1, 2)) -> Graph:
    a, b, c = ids
    return Graph.of({a: 0, b: 1, c: 2}, [(a, b, 0), (b, c, 0)])


def edge_ab() -> Graph:
    return Graph.of({0: 0, 1: 1}, [(0, 1, 0)])


def test_signature_is_isomorphism_invariant():
    # Same labeled path spelled with different node ids -> same signature.
    assert canonical_signature(path_abc((0, 1, 2))) == canonical_signature(path_abc((5, 7, 9)))
    # Undirected path C-B-A is the same graph as A-B-C.
    reversed_path = Graph.of({0: 2, 1: 1, 2: 0}, [(0, 1, 0), (1, 2, 0)])
    assert canonical_signature(reversed_path) == canonical_signature(path_abc())
    # A-B and B-A are the same undirected edge.
    assert canonical_signature(edge_ab()) == canonical_signature(Graph.of({0: 1, 1: 0}, [(0, 1, 0)]))


def test_signature_distinguishes_structure_and_labels():
    assert canonical_signature(edge_ab()) != canonical_signature(Graph.of({0: 0, 1: 2}, [(0, 1, 0)]))  # A-B vs A-C
    assert canonical_signature(edge_ab()) != canonical_signature(path_abc())                            # edge vs path
    # Same endpoints, different edge label.
    assert canonical_signature(edge_ab()) != canonical_signature(Graph.of({0: 0, 1: 1}, [(0, 1, 1)]))


def test_signature_matches_ground_truth():
    truth = json.loads(GROUND_TRUTH.read_text())
    by_id = {p["id"]: p for p in truth["patterns"]}
    assert canonical_signature(edge_ab()) == by_id["P1"]["signature"]
    assert canonical_signature(path_abc()) == by_id["P2"]["signature"]


def test_load_sample_dataset():
    graphs, labels = load_dataset(SAMPLE)
    assert labels == ["A", "B", "C"]
    assert len(graphs) == 3
    g0 = graphs[0]
    assert g0.num_nodes() == 3 and g0.num_edges() == 2
    assert g0.weight(0, 1) == 0.8 and g0.weight(1, 2) == 0.4
    assert g0.degree(1) == 2  # B is the middle node
    assert graphs[2].num_nodes() == 2  # third graph is a single A-B edge
    # Weights are undirected: querying either orientation returns the same value.
    assert g0.weight(1, 0) == g0.weight(0, 1)


def test_particle_signature_ignores_pivot():
    pattern = path_abc()
    p_at_b = Particle(pattern, pivot=1)
    p_at_a = Particle(pattern, pivot=0)
    assert p_at_b.signature() == p_at_a.signature() == canonical_signature(pattern)
    with pytest.raises(ValueError):
        Particle(pattern, pivot=99)
