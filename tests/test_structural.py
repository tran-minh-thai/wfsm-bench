"""Motif-context positions: the difference operator must see structure the template
multiset throws away.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

from pathlib import Path

import pytest

from wfsmbench.config import Params
from wfsmbench.core import Graph, Particle
from wfsmbench.core.signature import canonical_signature
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics
from wfsmbench.preprocess.template import EdgeTemplate
from wfsmbench.pso import optimize
from wfsmbench.pso.edits import Repivot, diff
from wfsmbench.pso.structural import (
    GrowFrom,
    ShrinkEdge,
    align,
    apply_structural,
    compose_structural,
    context_counter,
    diff_structural,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"

# Same three A-B edges, different wiring: a chain versus a star.
PATH = Graph.of({0: 0, 1: 1, 2: 0, 3: 1}, [(0, 1, 0), (1, 2, 0), (2, 3, 0)])
STAR = Graph.of({0: 1, 1: 0, 2: 0, 3: 0}, [(0, 1, 0), (0, 2, 0), (0, 3, 0)])


def test_the_template_difference_is_blind_to_topology():
    """The motivating defect: two different structures, no structural moves between them."""
    assert canonical_signature(PATH) != canonical_signature(STAR)
    moves = diff(Particle(PATH, 0), Particle(STAR, 0))
    assert not [m for m in moves if not isinstance(m, Repivot)]


def test_the_structural_difference_sees_it():
    moves = diff_structural(Particle(PATH, 0), Particle(STAR, 0))
    structural_moves = [m for m in moves if not isinstance(m, Repivot)]
    assert structural_moves, "the motif context must distinguish a path from a star"
    assert any(isinstance(m, GrowFrom) for m in structural_moves)
    assert any(isinstance(m, ShrinkEdge) for m in structural_moves)


def test_context_counter_separates_the_two_shapes():
    assert context_counter(PATH) != context_counter(STAR)
    # The star's three edges are interchangeable: each joins the degree-3 hub to a leaf.
    assert len(context_counter(STAR)) == 1
    # The chain has three distinguishable edges: both ends, and the middle one.
    assert len(context_counter(PATH)) == 3


def test_a_particle_is_its_own_target():
    assert diff_structural(Particle(PATH, 0), Particle(PATH, 0)) == []


def test_both_velocity_modes_search_and_stay_within_the_optimum():
    from wfsmbench.baselines import mine_exhaustive

    graphs, _ = load_dataset(SAMPLE)
    stats = build_statistics(graphs)
    best = mine_exhaustive(graphs, stats, Params()).gbest.fitness
    for mode in ("template", "structural"):
        params = Params(velocity_mode=mode)
        result = optimize(graphs, stats, params, seed=0)
        assert result.gbest is not None
        assert result.gbest.fitness <= best + 1e-12


def test_velocity_mode_is_validated():
    Params(velocity_mode="template"), Params(velocity_mode="structural")
    with pytest.raises(ValueError):
        Params(velocity_mode="motif")


def test_alignment_is_partial_not_all_or_nothing():
    """Requiring every source node to find a partner empties the map whenever the labels
    differ, and an empty map yields no growth moves at all -- the particle never leaves its
    seed. Overlapping labels must still align."""
    a = Graph.of({0: 0, 1: 1}, [(0, 1, 0)])
    b = Graph.of({0: 0, 1: 2}, [(0, 1, 0)])       # shares label A only
    mapping = align(a, b)
    assert mapping, "the shared label must still be matched"
    assert set(mapping) <= set(a.node_labels)
    # No label in common leaves nothing to align.
    assert align(a, Graph.of({0: 3, 1: 4}, [(0, 1, 0)])) == {}


def test_a_seed_particle_can_still_grow_toward_a_differently_labelled_target():
    """The regression that made every structural move fail: two one-edge patterns with
    different labels produced only shrinks, which a one-edge pattern cannot perform."""
    from wfsmbench.pso.structural import apply_structural

    source = Particle(Graph.of({0: 0, 1: 1}, [(0, 1, 0)]), 0)
    target = Particle(Graph.of({0: 1, 1: 2}, [(0, 1, 0)]), 0)
    moves = diff_structural(source, target)
    grows = [m for m in moves if isinstance(m, GrowFrom)]
    assert grows, "there must be a way to move toward the target"
    assert any(apply_structural(source, m) is not None for m in grows)
