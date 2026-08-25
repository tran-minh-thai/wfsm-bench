"""P6 - graph-edit moves, diff (Delta) and apply_velocity (Phi).

Label encoding: A=0, B=1, C=2; edge label 0.
"""

from wfsmbench.config import Params
from wfsmbench.core import Graph, Particle, canonical_signature
from wfsmbench.preprocess import EdgeTemplate
from wfsmbench.pso import Grow, Repivot, Shrink, apply_move, apply_velocity, diff, is_valid

AB_T = EdgeTemplate.of(0, 1, 0)   # A-B
BC_T = EdgeTemplate.of(1, 2, 0)   # B-C
AC_T = EdgeTemplate.of(0, 2, 0)   # A-C
FREQUENT = {AB_T, BC_T, AC_T}


def edge_ab():
    return Particle(Graph.of({0: 0, 1: 1}, [(0, 1, 0)]), pivot=1)          # A-B, pivot B


def path_abc():
    return Particle(Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)]), pivot=1)  # pivot B


def sig(particle):
    return canonical_signature(particle.pattern)


def test_grow_expands_from_pivot():
    grown = apply_move(edge_ab(), Grow(BC_T))
    assert sig(grown) == sig(path_abc())     # A-B grown at pivot B into path A-B-C
    assert grown.pattern.num_edges() == 2


def test_grow_closes_a_cycle():
    triangle = apply_move(path_abc(), Grow(AC_T))
    assert triangle.pattern.num_edges() == 3
    assert triangle.pattern.edge(0, 2) is not None   # the closing A-C edge


def test_shrink_removes_edge_and_isolated_node():
    shrunk = apply_move(path_abc(), Shrink(BC_T))
    assert sig(shrunk) == sig(edge_ab())     # back to A-B, C dropped
    assert shrunk.pivot == 1                  # B still present


def test_repivot():
    moved = apply_move(path_abc(), Repivot(2))   # to the C node
    assert moved.pivot == 2
    assert sig(moved) == sig(path_abc())         # structure unchanged
    assert apply_move(path_abc(), Repivot(1)) is None  # already at a B node -> no-op


def test_diff_then_apply_reaches_target_structure():
    x, y = edge_ab(), path_abc()
    moves = diff(x, y)
    assert moves == [Grow(BC_T)]
    reached = apply_velocity(x, moves, FREQUENT, Params())
    assert sig(reached) == sig(y)


def test_is_valid():
    params = Params()
    assert is_valid(path_abc(), FREQUENT, params)
    # Disconnected pattern.
    split = Particle(Graph.of({0: 0, 1: 1, 2: 2, 3: 0}, [(0, 1, 0), (2, 3, 0)]), pivot=0)
    assert not is_valid(split, FREQUENT, params)
    # Over the node budget.
    assert not is_valid(path_abc(), FREQUENT, Params(max_nodes=2))
    # An edge whose template is not frequent.
    assert not is_valid(path_abc(), {AB_T}, params)
    # No edges.
    assert not is_valid(Particle(Graph.of({0: 0}, []), pivot=0), FREQUENT, params)


def test_apply_velocity_skips_moves_that_break_validity():
    params = Params(max_nodes=3)
    # Second Grow would need a 4th node; it must be skipped, leaving the 3-node path.
    result = apply_velocity(edge_ab(), [Grow(BC_T), Grow(BC_T)], FREQUENT, params)
    assert result.pattern.num_nodes() == 3
