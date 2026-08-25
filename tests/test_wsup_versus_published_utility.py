"""WSUP against the published utility measure it most resembles.

UGMINE (Alam et al., Applied Intelligence 53:49-68) mines edge-weighted graph transaction
databases -- the same setting used here -- and defines, in its Eqs. (14)-(16),

    u_g(g)      = SUM over edges of the embedding of the edge utility
    u_G(g, G)   = MAX of u_g over the embeddings of g in host graph G
    u_D(g, D)   = SUM of u_G over the host graphs

while the measure used here is

    ATW(g)      = MEAN of the embedding weight over every embedding in the database
    WSUP(g)     = supp(g) * ATW(g),  supp = number of host graphs with an embedding

These are not the same aggregation, and the claim in the survey is the precise one: they
coincide exactly when every supporting graph carries a single embedding, and diverge
otherwise. Both halves are checked here, because a claim of equality that is never shown to
fail is not a claim about anything. The reference implementation is written from UGMINE's
published equations, not from this project's code, so that the comparison recomputes rather
than restates.
"""
from __future__ import annotations

import math

from wfsmbench.measure.evaluation import Evaluation


def ugmine_utility(embeddings_by_graph: dict[int, list[list[float]]]) -> float:
    """u_D(g, D) from UGMINE Eqs. (14)-(16): sum over graphs of the max embedding sum."""
    total = 0.0
    for embeddings in embeddings_by_graph.values():
        if embeddings:
            total += max(sum(weights) for weights in embeddings)
    return total


def test_identical_when_each_supporting_graph_has_one_embedding() -> None:
    single = {0: [[1.0, 2.0]], 1: [[4.0]], 2: [], 3: [[0.5, 0.5, 3.0]]}
    evaluation = Evaluation.from_embeddings(single)
    assert evaluation.support == 3, "support counts host graphs, not embeddings"
    assert math.isclose(evaluation.wsup, ugmine_utility(single)), (
        f"WSUP {evaluation.wsup} should equal UGMINE u_D {ugmine_utility(single)} "
        "when every supporting graph has exactly one embedding"
    )


def test_diverges_once_a_graph_carries_several_embeddings() -> None:
    """The equality must be shown to break, or it is not a statement about the aggregation.

    The multi-embedding graph carries two *distinct non-zero* weights on purpose. An earlier
    version used [[10.0], [0.0]], where max and sum coincide, so replacing UGMINE's max by a
    sum left every assertion passing -- the case could not tell the two aggregations apart.
    """
    many = {0: [[10.0], [6.0]], 1: [[4.0]]}
    evaluation = Evaluation.from_embeddings(many)
    # WSUP: supp 2, embeddings [10, 6, 4], mean 20/3 -> 2 * 20/3
    assert math.isclose(evaluation.wsup, 2 * (20.0 / 3.0))
    # UGMINE: max(10, 6) + 4 = 14, and a sum-instead-of-max reading would give 20
    assert math.isclose(ugmine_utility(many), 14.0), (
        "the reference must take the per-graph maximum, not the per-graph sum"
    )
    assert not math.isclose(evaluation.wsup, ugmine_utility(many)), (
        "the two measures must differ here; if they agree the divergence claim is wrong"
    )


def test_empty_and_unsupported_cases_agree_at_zero() -> None:
    for case in ({}, {0: [], 1: []}):
        evaluation = Evaluation.from_embeddings(case)
        assert evaluation.support == 0
        assert evaluation.wsup == 0.0 == ugmine_utility(case)
