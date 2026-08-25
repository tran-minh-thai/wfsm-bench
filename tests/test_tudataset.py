"""P1 - TUDataset loader with edge labels + edge weights.

Two layers: a hand-made fixture that locks the parsing logic offline, and a check
against the paper's Table 2 statistics when the real raw datasets are present.
"""

from pathlib import Path

import pytest

from wfsmbench.io.tudataset import convert, load_graphs

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _write_fixture(root: Path) -> Path:
    name = "FIX"
    d = root / name
    d.mkdir(parents=True)
    (d / f"{name}_graph_indicator.txt").write_text("1\n1\n1\n2\n2\n")
    (d / f"{name}_node_labels.txt").write_text("0\n0\n1\n1\n0\n")
    # Undirected edges 1-2, 2-3 (graph 1) and 4-5 (graph 2), each listed both ways.
    (d / f"{name}_A.txt").write_text("1, 2\n2, 1\n2, 3\n3, 2\n4, 5\n5, 4\n")
    (d / f"{name}_edge_labels.txt").write_text("0\n0\n1\n1\n0\n0\n")
    (d / f"{name}_edge_attributes.txt").write_text("2.0\n2.0\n4.0\n4.0\n6.0\n6.0\n")
    return root


def test_fixture_parsing_and_weight_scaling(tmp_path):
    _write_fixture(tmp_path)
    data = convert(tmp_path, "FIX")

    assert data["label_strings"] == ["0", "1"]
    assert len(data["graphs"]) == 2

    g0, g1 = data["graphs"]
    # Graph 1: 3 nodes, 2 undirected edges (deduplicated from 4 directed lines).
    assert len(g0["vertices"]) == 3 and len(g0["edges"]) == 2
    edges = {(e["src"], e["dst"]): (e["label"], e["weight"]) for e in g0["edges"]}
    # d=2 -> 0.5 (min), d=4 -> 2.75 (mid). Range is [2,6] -> [0.5,5.0].
    assert edges[(0, 1)] == (0, 0.5)
    assert edges[(1, 2)] == (1, 2.75)
    # Graph 2: single edge, d=6 -> 5.0 (max).
    assert len(g1["edges"]) == 1
    assert g1["edges"][0]["weight"] == 5.0


def test_fixture_load_graphs(tmp_path):
    _write_fixture(tmp_path)
    graphs = load_graphs(tmp_path, "FIX")
    assert len(graphs) == 2
    assert graphs[0].num_nodes() == 3 and graphs[0].num_edges() == 2
    assert graphs[0].degree(1) == 2  # middle node of the path


@pytest.mark.skipif(not (RAW / "BZR_MD").is_dir(), reason="raw BZR_MD not downloaded")
def test_bzr_md_matches_table2():
    data = convert(RAW, "BZR_MD")
    graphs = data["graphs"]
    n = len(graphs)
    avg_nodes = sum(len(g["vertices"]) for g in graphs) / n
    avg_edges = sum(len(g["edges"]) for g in graphs) / n
    weights = [e["weight"] for g in graphs for e in g["edges"]]
    avg_w = sum(weights) / len(weights)

    assert n == 306
    assert avg_nodes == pytest.approx(21.304, abs=1e-3)
    assert avg_edges == pytest.approx(225.056, abs=1e-3)
    # D7: min-max scaling reproduces the paper's average weight exactly.
    assert avg_w == pytest.approx(1.519271, abs=1e-5)
    assert min(weights) == pytest.approx(0.5) and max(weights) == pytest.approx(5.0)


@pytest.mark.skipif(not (RAW / "Cuneiform").is_dir(), reason="raw Cuneiform not downloaded")
def test_cuneiform_structure_matches_table2():
    data = convert(RAW, "Cuneiform")
    graphs = data["graphs"]
    n = len(graphs)
    avg_nodes = sum(len(g["vertices"]) for g in graphs) / n
    avg_edges = sum(len(g["edges"]) for g in graphs) / n
    weights = [e["weight"] for g in graphs for e in g["edges"]]

    assert n == 267
    assert avg_nodes == pytest.approx(21.273, abs=1e-3)
    assert avg_edges == pytest.approx(44.798, abs=1e-3)
    # Weights stay in range (paper's exact avg is not reproducible for Cuneiform - see D7).
    assert min(weights) >= 0.5 and max(weights) <= 5.0
