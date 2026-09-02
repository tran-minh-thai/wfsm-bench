"""Rebuilding TUDataset's molecular collections into weighted sparse graphs.

The parts that can go wrong quietly are pinned here: that only real bonds become edges (the
whole reason this exists rather than using the _MD collections), that the weight really is the
distance between the bonded atoms, and that the two directions the raw file lists for one bond
do not become two edges.
"""

import json
from pathlib import Path

import pytest

from wfsmbench.io.molecular import LAYOUTS, build_graphs, convert, load_graphs
from wfsmbench.io.tudataset import WEIGHT_MAX, WEIGHT_MIN

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Two molecules. The first is a path A-B-C, the second a single bond A-B. Coordinates are
# chosen so the bond lengths are 1, 2 and 4 -- distinct, and easy to check by eye.
GRAPH_OF_NODE = [1, 1, 1, 2, 2]
NODE_LABELS = [6, 7, 6, 6, 8]
POSITIONS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0),
             (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
EDGE_PAIRS = [(1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4)]   # both directions, as raw


def build():
    return build_graphs(GRAPH_OF_NODE, NODE_LABELS, POSITIONS, EDGE_PAIRS)


def test_only_real_bonds_become_edges():
    """The point of the rebuild. The _MD collections join every pair of atoms, which makes
    the graphs complete and support near-constant; here node 1 and node 3 are two bonds apart
    and must not be joined."""
    graphs, _labels = build()
    first = graphs[0]
    assert len(first["vertices"]) == 3
    assert len(first["edges"]) == 2                       # not 3, which is what complete is
    assert {(e["src"], e["dst"]) for e in first["edges"]} == {(0, 1), (1, 2)}


def test_both_raw_directions_make_one_edge():
    graphs, _labels = build()
    assert len(graphs[1]["edges"]) == 1


def test_weight_is_the_distance_between_the_bonded_atoms():
    """Lengths 1, 2 and 4 min-max scale to the bottom, the middle third and the top."""
    graphs, _labels = build()
    by_length = {(e["src"], e["dst"]): e["weight"] for e in graphs[0]["edges"]}
    assert by_length[(0, 1)] == pytest.approx(WEIGHT_MIN)               # 1.0, the shortest
    assert by_length[(1, 2)] == pytest.approx(
        WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * (2.0 - 1.0) / (4.0 - 1.0))
    assert graphs[1]["edges"][0]["weight"] == pytest.approx(WEIGHT_MAX)  # 4.0, the longest


def test_node_labels_are_remapped_contiguously():
    graphs, labels = build()
    assert labels == [6, 7, 8]
    assert [v["label"] for v in graphs[0]["vertices"]] == [0, 1, 0]
    assert [v["label"] for v in graphs[1]["vertices"]] == [0, 2]


def test_edge_labels_default_to_a_constant():
    """The collections without bond types have to assign one, and it must never come from the
    bond length -- that is already the weight, and deriving the label from it would let the
    weight decide what matches what."""
    graphs, _labels = build()
    assert {e["label"] for g in graphs for e in g["edges"]} == {0}


def test_edge_labels_are_read_from_the_one_hot_when_present():
    one_hot_rows = [1, 1, 0, 0, 3, 3]     # already argmaxed, one per raw _A.txt line
    graphs, _labels = build_graphs(GRAPH_OF_NODE, NODE_LABELS, POSITIONS, EDGE_PAIRS,
                                   edge_labels=one_hot_rows)
    assert [e["label"] for e in graphs[0]["edges"]] == [1, 0]
    assert [e["label"] for e in graphs[1]["edges"]] == [3]


def test_limit_selects_a_prefix_of_the_molecules():
    """SUA 2026-08-27: check WHICH molecules come back, not only how many.

    The previous version asserted len(graphs) == 1 after asking for one molecule, which any
    selection rule satisfies -- the last molecule, a random one, an empty one rebuilt from
    nothing. Strengthening it to compare the molecule itself turned up something the count
    could never have shown, and it is not a bug: see the test below.
    """
    tat_ca, _labels = build()
    graphs, _labels = build_graphs(GRAPH_OF_NODE, NODE_LABELS, POSITIONS, EDGE_PAIRS, limit=1)
    assert len(graphs) == 1
    assert tat_ca[0] != tat_ca[1], "both molecules are identical; this check cannot fail"

    # Structure, not weights: the weights are the subject of the next test.
    def hinh_dang(g):
        return (g["id"], g["vertices"],
                sorted((e["src"], e["dst"], e["label"]) for e in g["edges"]))

    assert hinh_dang(graphs[0]) == hinh_dang(tat_ca[0]), (
        f"limit=1 returned a molecule that is not the first: got {hinh_dang(graphs[0])}, "
        f"the collection starts with {hinh_dang(tat_ca[0])}")


def test_the_weight_scale_is_a_property_of_the_subset_that_was_loaded():
    """Loading fewer molecules changes the weights of the ones that remain.

    Found by strengthening the check above. Weights are min-max scaled to [WEIGHT_MIN,
    WEIGHT_MAX] against the anchors of whatever was loaded, so the SAME bond gets a different
    weight in a different subset. Here the bond lengths are 1, 2 and 4: loading both molecules
    scales the second bond of the first molecule to 2.0, loading only the first scales that
    same bond to WEIGHT_MAX, because within that subset it is the longest bond there is.

    This is pinned rather than fixed, for two reasons. It is the documented behaviour of the
    loader, and the manuscript states the consequence directly: on the subsets actually used,
    the w_max anchor is not the nominal maximum but runs from 3.475 to 5.000 by collection and
    subset size. Anyone making the scaling global would make that sentence false, and this is
    what would tell them.
    """
    tat_ca, _labels = build()
    mot, _labels = build_graphs(GRAPH_OF_NODE, NODE_LABELS, POSITIONS, EDGE_PAIRS, limit=1)

    def trong_so(g):
        return [e["weight"] for e in sorted(g["edges"], key=lambda e: (e["src"], e["dst"]))]

    day_du, cat_bot = trong_so(tat_ca[0]), trong_so(mot[0])
    assert day_du != cat_bot, (
        f"the same molecule kept its weights across two different subsets ({day_du}); the "
        f"scaling has become global, and the manuscript's note about the w_max anchor "
        f"varying by subset is now false")
    assert min(day_du) == min(cat_bot) == WEIGHT_MIN, "the shortest bond is the floor in both"
    assert max(cat_bot) == WEIGHT_MAX, (
        f"in a subset holding only the first molecule its longest bond must reach the ceiling; "
        f"got {cat_bot}")
    assert max(day_du) < WEIGHT_MAX, (
        f"with the longer second molecule loaded, the first molecule's longest bond is no "
        f"longer the longest overall and must sit below the ceiling; got {day_du}")

def test_a_collection_without_a_role_assignment_is_refused():
    with pytest.raises(ValueError):
        convert(RAW, "MUTAG")


@pytest.mark.parametrize("name", sorted(LAYOUTS))
@pytest.mark.skipif(not RAW.is_dir(), reason="raw TUDataset files not downloaded")
def test_rebuilt_collections_are_sparse_and_chemically_plausible(name):
    """Against the real data when it is present. The _MD collections this replaces run at
    E/N above 10; a molecular bond graph is barely above 1, and its weights have to span the
    scaled range rather than collapse."""
    if not (RAW / name).is_dir():
        pytest.skip(f"{name} not downloaded")
    graphs = load_graphs(RAW, name, limit=50)
    nodes = sum(g.num_nodes() for g in graphs)
    edges = sum(g.num_edges() for g in graphs)
    assert 0.8 < edges / nodes < 1.5, f"{name} is not sparse: E/N = {edges / nodes:.2f}"
    weights = [w for g in graphs for (_label, w) in g.edges.values()]
    assert min(weights) >= WEIGHT_MIN - 1e-9
    assert max(weights) <= WEIGHT_MAX + 1e-9
