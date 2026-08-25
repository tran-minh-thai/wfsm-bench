"""P3 - weight measures (W, ATW, WSUP) and fitness F, locked on the sample.

ATW here is this project's definition (mean over embeddings of the summed matched edge
weights), which differs from other WFSM measures. Ground truth is hand-computed in
data/sample/ground_truth.json.
"""

import json
from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.measure import (Evaluation, discounted_weight, embedding_weight,
                               fitness, norm, penalty)
from wfsmbench.preprocess import build_statistics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
GROUND_TRUTH = Path(__file__).resolve().parents[1] / "data" / "sample" / "ground_truth.json"

# Embeddings that the matcher (P4) will later produce for the two anchor patterns.
P1_EMBEDDINGS = {0: [[0.8]], 1: [[0.6]], 2: [[1.0]]}      # single edge A-B
P2_EMBEDDINGS = {0: [[0.7, 0.5]], 1: [[0.6, 0.6]]}        # path A-B-C, per-edge weights


def sample_stats():
    graphs, _ = load_dataset(SAMPLE)
    return build_statistics(graphs)


def ground_truth():
    return {p["id"]: p for p in json.loads(GROUND_TRUTH.read_text())["patterns"]}


def test_embedding_weight_is_a_sum():
    assert embedding_weight([0.8]) == 0.8
    assert embedding_weight([0.7, 0.5]) == pytest.approx(1.2)


def test_evaluation_matches_ground_truth():
    truth = ground_truth()
    p1 = Evaluation.from_embeddings(P1_EMBEDDINGS)
    assert p1.support == truth["P1"]["support"]
    assert p1.atw == pytest.approx(truth["P1"]["atw"])
    assert p1.wsup == pytest.approx(truth["P1"]["wsup"])
    p2 = Evaluation.from_embeddings(P2_EMBEDDINGS)
    assert p2.support == truth["P2"]["support"]
    assert p2.atw == pytest.approx(truth["P2"]["atw"])
    assert p2.wsup == pytest.approx(truth["P2"]["wsup"])


def test_evaluation_empty():
    e = Evaluation.from_embeddings({})
    assert (e.support, e.atw, e.wsup) == (0, 0.0, 0.0)


def test_norm_and_penalty():
    assert norm(2.0, 4.0) == 0.5
    assert norm(10.0, 5.0) == 1.0     # clipped above
    assert norm(-1.0, 5.0) == 0.0     # clipped below
    assert norm(1.0, 0.0) == 0.0      # zero reference
    params = Params()
    assert penalty(2, 1, params) == pytest.approx(0.3)   # 0.1*2 + 0.1*1


def test_fitness_hand_computed():
    stats = sample_stats()   # max_edge_weight=1.0, num_graphs=3
    params = Params()        # defaults (D3); size_reference=6 -> atw_ref=6, wsup_ref=18
    f1 = fitness(Evaluation.from_embeddings(P1_EMBEDDINGS), num_nodes=2, num_edges=1,
                 stats=stats, params=params)
    f2 = fitness(Evaluation.from_embeddings(P2_EMBEDDINGS), num_nodes=3, num_edges=2,
                 stats=stats, params=params)
    # F(P1) = 1*(2.4/18) + 0.5*(0.8/6) - 0.1*(0.3) = 0.17
    assert f1 == pytest.approx(0.17)
    # F(P2) = 1*(2.4/18) + 0.5*(1.2/6) - 0.1*(0.5) = 0.183333...
    assert f2 == pytest.approx(0.1833333, abs=1e-6)


def test_fitness_does_not_depend_on_the_search_bound():
    """F(g) has to be a function of the pattern and the dataset alone.

    The anchor used to be scaled by params.max_edges, which bounds the search space, so the
    same pattern scored differently under different bounds: on BZR_MD the exhaustive optimum
    came out at 0.6560 for max_size 3 and 0.6478 for max_size 4, which cannot happen when
    every size-3 pattern is still feasible at size 4. The optimum had not moved -- the
    denominator had. Holding size_reference fixed is what makes a size sweep comparable.
    """
    stats = sample_stats()
    evaluation = Evaluation.from_embeddings(P2_EMBEDDINGS)   # 3 nodes, 2 edges
    scores = {
        fitness(evaluation, num_nodes=3, num_edges=2, stats=stats,
                params=Params(max_nodes=bound, max_edges=bound, size_reference=6))
        for bound in (2, 3, 4, 5, 6)
    }
    assert len(scores) == 1, f"fitness moved with the search bound: {sorted(scores)}"


def test_size_reference_must_cover_the_search_bound():
    """An anchor below the bound clips the heaviest patterns to Norm = 1, which flattens
    them into a tie and leaves the penalty term deciding gbest. Reject it outright."""
    Params(max_edges=6, size_reference=6)
    Params(max_edges=4, size_reference=6)
    with pytest.raises(ValueError):
        Params(max_edges=8, size_reference=6)


def test_fitness_per_edge_mode_is_size_neutral():
    stats = sample_stats()   # max_edge_weight=1.0, num_graphs=3
    params = Params(weight_mode="per_edge")
    # per_edge divides W(e) by the edge count, so ATW is the average weight per edge and
    # the anchors drop the size_reference factor (atw_ref = max_edge_weight = 1.0).
    e2 = Evaluation.from_embeddings(P2_EMBEDDINGS)   # atw=1.2 over 2 edges, support=2
    f2 = fitness(e2, num_nodes=3, num_edges=2, stats=stats, params=params)
    # atw_pe = 1.2/2 = 0.6; wsup_pe = 2*0.6 = 1.2; refs atw_ref=1.0, wsup_ref=3*1.0=3.
    # F = 1*(1.2/3) + 0.5*(0.6/1.0) - 0.1*(0.1*3 + 0.1*2) = 0.4 + 0.3 - 0.05 = 0.65
    assert f2 == pytest.approx(0.65, abs=1e-6)


def test_discounted_weight_saturates_with_size():
    """The point of the mode: W(e) is bounded independently of how many edges there are.

    A plain sum of k equal weights grows without limit in k; the discounted sum converges to
    W_max/(1-rho). That bound is what makes the normalized ATW saturate, which is the
    condition an interior optimum needs (scripts/run_degeneracy.py).
    """
    aggregate = discounted_weight(0.5)
    assert aggregate([1.0]) == pytest.approx(1.0)
    assert aggregate([1.0, 1.0]) == pytest.approx(1.5)          # 1 + 0.5
    assert aggregate([1.0] * 8) == pytest.approx(1.9921875)     # -> 2.0 = 1/(1-0.5)
    assert aggregate([1.0] * 40) < 2.0
    # Heaviest first, so the order it is handed in cannot change the answer.
    assert aggregate([0.2, 0.9, 0.5]) == pytest.approx(0.9 + 0.5 * 0.5 + 0.2 * 0.25)


def test_discounted_at_rho_one_is_the_plain_sum():
    """rho = 1 discounts nothing, so the new mode has to reproduce "sum" exactly -- both in
    the aggregation and in the normalization anchor that goes with it."""
    stats = sample_stats()
    assert discounted_weight(1.0)([0.7, 0.5]) == pytest.approx(embedding_weight([0.7, 0.5]))
    assert (stats.atw_reference_discounted(6, rho=1.0)
            == pytest.approx(stats.atw_reference(6)))
    evaluation = Evaluation.from_embeddings(P2_EMBEDDINGS, discounted_weight(1.0))
    plain = Evaluation.from_embeddings(P2_EMBEDDINGS)
    assert evaluation == plain
    assert (fitness(evaluation, 3, 2, stats, Params(weight_mode="discounted", rho=1.0))
            == pytest.approx(fitness(plain, 3, 2, stats, Params(weight_mode="sum"))))


def test_weight_mode_is_validated():
    Params(weight_mode="sum"), Params(weight_mode="per_edge")
    Params(weight_mode="discounted", rho=0.6)
    with pytest.raises(ValueError):
        Params(weight_mode="mean")
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Params(rho=bad)


def test_params_from_dict_rejects_unknown():
    assert Params.from_dict({"alpha": 2.0}).alpha == 2.0
    with pytest.raises(ValueError):
        Params.from_dict({"nope": 1})
