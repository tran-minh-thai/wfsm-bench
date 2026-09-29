"""Discrete PSO operators and the full search loop.

Label encoding: A=0, B=1, C=2; edge label 0.
"""

from pathlib import Path

import numpy as np
import pytest

from wfsmbench.config import Params
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import EdgeTemplate, build_statistics
from wfsmbench.pso import Grow, Repivot, Shrink, compose, is_valid, optimize, scale

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "illustrative.json"
AB_T = EdgeTemplate.of(0, 1, 0)
BC_T = EdgeTemplate.of(1, 2, 0)
AB_SIG = "V[0,1]E[0-1:0]"


def sample_setup():
    graphs, _ = load_dataset(SAMPLE)
    stats = build_statistics(graphs)
    return graphs, stats


def test_scale_keeps_all_or_none():
    moves = [Grow(AB_T)] * 10
    assert scale(moves, 0.0, np.random.default_rng(0)) == []
    assert scale(moves, 1.0, np.random.default_rng(0)) == moves
    partial = scale(moves, 0.5, np.random.default_rng(0))
    assert 0 < len(partial) < 10


def test_compose_cancels_and_caps():
    # A Grow and a Shrink of the same template cancel out.
    assert compose([Grow(AB_T)], [Shrink(AB_T)], [], max_velocity=3) == []
    # Length is capped after resolution (templates emitted in canonical order).
    assert compose([Grow(AB_T), Grow(BC_T)], [], [], max_velocity=1) == [Grow(AB_T)]
    # Exactly one Repivot survives - the social (last) one.
    assert compose([Repivot(1)], [Repivot(2)], [], max_velocity=3) == [Repivot(2)]


def test_optimize_is_deterministic_for_a_seed():
    graphs, stats = sample_setup()
    params = Params()
    a = optimize(graphs, stats, params, seed=0)
    b = optimize(graphs, stats, params, seed=0)
    assert a.gbest.signature == b.gbest.signature
    assert a.gbest.fitness == b.gbest.fitness
    assert [(c.signature, c.fitness) for c in a.top_k] == [(c.signature, c.fitness) for c in b.top_k]


def test_optimize_finds_frequent_weighted_patterns():
    graphs, stats = sample_setup()
    params = Params()
    frequent = set(stats.frequent_edge_templates(params.min_support))
    result = optimize(graphs, stats, params, seed=0)

    assert result.gbest is not None
    assert len(result.top_k) >= 1
    # Every archived candidate is a valid, frequent pattern; signatures are distinct.
    sigs = [c.signature for c in result.top_k]
    assert len(sigs) == len(set(sigs))
    for c in result.top_k:
        assert is_valid(c.particle, frequent, params)
        assert c.evaluation.support >= params.min_support
    # gbest is the best of the archive, and at least as good as the A-B seed (F=0.17).
    assert result.gbest.fitness == pytest.approx(max(c.fitness for c in result.top_k))
    assert result.gbest.fitness >= 0.17 - 1e-9
    assert AB_SIG in sigs


def test_optimize_with_no_frequent_templates_returns_empty():
    graphs, stats = sample_setup()
    result = optimize(graphs, stats, Params(min_support=99), seed=0)
    assert result.top_k == [] and result.gbest is None


def test_compose_puts_shrinks_before_grows():
    """A grow is rejected when the pattern is already at its size bound, so removals must
    come first or the whole velocity is silently discarded."""
    velocity = compose([Grow(BC_T)], [Shrink(AB_T)], [], max_velocity=3)
    assert velocity == [Shrink(AB_T), Grow(BC_T)]


def test_compose_keeps_the_pivot_move_within_the_cap():
    """The pivot move is placed first, so truncation cannot drop the pivot mechanism, and
    it applies before the grows that expand from the pivot."""
    velocity = compose([Grow(AB_T), Grow(BC_T)], [], [Repivot(2)], max_velocity=2)
    assert velocity[0] == Repivot(2)
    assert len(velocity) == 2
    # Without a cap the pivot still leads, followed by the structural moves.
    full = compose([Grow(AB_T)], [], [Repivot(2)], max_velocity=5)
    assert full == [Repivot(2), Grow(AB_T)]


def test_compose_truncates_by_priority_not_template_id():
    """Inertia contributes first, so its template survives a tight cap even when another
    part contributes a template that sorts earlier."""
    # BC_T sorts after AB_T, but inertia (BC_T) outranks the social pull (AB_T).
    velocity = compose([Grow(BC_T)], [], [Grow(AB_T)], max_velocity=1)
    assert velocity == [Grow(BC_T)]


def test_a_repivot_move_persists_through_an_iteration():
    """The pivot is part of a particle's position: a Repivot must still be in effect after
    the move is applied, not overwritten by re-scoring the pivot from the pattern."""
    from wfsmbench.core import Graph, Particle
    from wfsmbench.preprocess.template import EdgeTemplate
    from wfsmbench.pso.edits import apply_velocity

    pattern = Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)])
    frequent = {EdgeTemplate.of(0, 1, 0), EdgeTemplate.of(1, 2, 0)}
    start_particle = Particle(pattern, 0)
    moved = apply_velocity(start_particle, [Repivot(2)], frequent, Params())
    assert moved.pivot != start_particle.pivot
    assert moved.pattern.label(moved.pivot) == 2      # the pivot really moved to label C


def test_the_evaluator_charges_only_for_new_patterns():
    """Budgets count distinct patterns scored: revisiting a cached one is free, so a search
    that wanders back over old ground is not penalised twice for it."""
    from wfsmbench.core import Graph, Particle
    from wfsmbench.io import load_dataset
    from wfsmbench.preprocess import build_statistics
    from wfsmbench.pso.optimizer import Evaluator

    graphs, _ = load_dataset(SAMPLE)
    stats = build_statistics(graphs)
    evaluator = Evaluator(graphs, stats, Params(), budget=2)

    particle = Particle(Graph.of({0: 0, 1: 1}, [(0, 1, 0)]), 0)
    first = evaluator.score(particle)
    assert evaluator.spent == 1
    again = evaluator.score(particle)                  # same pattern, no new cost
    assert evaluator.spent == 1
    assert again.fitness == first.fitness
    assert not evaluator.exhausted

    evaluator.score(Particle(Graph.of({0: 0, 1: 1, 2: 2}, [(0, 1, 0), (1, 2, 0)]), 0))
    assert evaluator.spent == 2 and evaluator.exhausted


def test_the_initial_swarm_depends_on_the_seed():
    """Taking the frequent templates in sorted order gave every run the same starting swarm,
    so runs differed only in what happened after the first iteration."""
    from wfsmbench.io import load_dataset
    from wfsmbench.preprocess import build_statistics

    graphs, _ = load_dataset(SAMPLE)
    stats = build_statistics(graphs)
    # iterations=0 spends the whole budget on seeding, so the reported set IS the initial
    # swarm; with one particle and several frequent templates the pick must vary by seed.
    params = Params(swarm_size=1, iterations=0)
    starts = {tuple(c.signature for c in optimize(graphs, stats, params, s).top_k)
              for s in range(8)}
    assert len(starts) > 1, "different seeds must start from different templates"
