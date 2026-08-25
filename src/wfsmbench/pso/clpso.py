"""Comprehensive-Learning PSO (CLPSO) over the same discrete graph-edit space.

CLPSO (Liang, Qin, Suganthan & Baskar, 2006) is a stronger PSO variant for multimodal
problems. It drops the single global-best attractor that makes the basic swarm converge
early, and instead has each particle learn, per dimension, from a *comprehensive* exemplar
assembled from the whole swarm's personal bests. Different dimensions may follow different
particles, which keeps the population diverse and better able to escape local optima.

The mapping to this problem treats each edge-template as a dimension. For a particle, each
template's learning target is taken from its own personal best, or -- with the particle's
CLPSO learning probability ``Pc`` -- from the better of two randomly chosen personal bests
(tournament). The velocity is the edit sequence that moves the particle toward that mixed
template target; there is no global-best pull. Everything else (fitness, archive, validity,
Top-K admission) is shared with :mod:`wfsmbench.pso.optimizer`, so a comparison against the
basic swarm and the baselines stays fair -- only the learning rule differs.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..core.signature import canonical_signature
from ..preprocess.statistics import DatasetStats
from .edits import Grow, Move, Shrink, _template_counter, apply_velocity, is_valid
from .optimizer import (
    Candidate,
    Result,
    _neighbors,
    reported_best,
    _seed_particle,
    compose,
    evaluate,
    STALL_ROUNDS,
    fresh_particles,
    make_evaluator,
    restart_count,
    scale,
    select_top_k,
)


def learning_probabilities(swarm_size: int) -> list[float]:
    """CLPSO per-particle learning probability Pc_i, spread over [0.05, 0.5] across the
    swarm so particles differ in how much they learn from others (Liang et al., eq. 4)."""
    if swarm_size <= 1:
        return [0.0]
    denominator = math.exp(10.0) - 1.0
    return [0.05 + 0.45 * (math.exp(10.0 * i / (swarm_size - 1)) - 1.0) / denominator
            for i in range(swarm_size)]


def _exemplar(pbests: list[Candidate], index: int,
              pc: float, rng: np.random.Generator) -> Counter:
    """The edits that move ``current`` toward its comprehensive exemplar.

    CLPSO learns each *dimension* from a possibly different exemplar. In the continuous
    setting every particle holds a value in every dimension, so a composite exemplar is
    always well defined. Graph patterns are sparse in template space -- a particle carries a
    handful of the templates that exist -- so asking each exemplar "how many of template t do
    you have?" answers zero almost every time, and the composite collapses to nothing.
    Measured on Cuneiform, that produced target patterns of 1.2 edges against a bound of 4
    and 0.3 growth moves per update: the swarm could not grow at all.

    The dimensions here are therefore the target's *edge slots*, not the whole template
    vocabulary. The exemplar's size comes from the particle's own best, and each slot is
    filled from that best, or -- with probability ``pc`` -- by copying a template from the
    better of two randomly drawn bests. Every slot contributes a real edge, so the composite
    is a pattern the particle can actually be pulled toward.
    """
    own = pbests[index]
    own_templates = list(_template_counter(own.particle.pattern).elements())
    if not own_templates:
        return Counter()

    def donor() -> Candidate:
        if len(pbests) > 1 and rng.random() < pc:
            a, b = int(rng.integers(len(pbests))), int(rng.integers(len(pbests)))
            return pbests[a] if pbests[a].fitness >= pbests[b].fitness else pbests[b]
        return own

    target: Counter = Counter()
    learned_from_other = False
    for _ in range(len(own_templates)):
        source = donor()
        learned_from_other = learned_from_other or source is not own
        options = list(_template_counter(source.particle.pattern).elements())
        if not options:
            options = own_templates
        target[options[int(rng.integers(len(options)))]] += 1

    # CLPSO requires learning from at least one other particle; if every draw fell back to
    # the own best, swap one slot for a template taken from another.
    if not learned_from_other and len(pbests) > 1:
        others = [j for j in range(len(pbests)) if j != index]
        donated = list(_template_counter(
            pbests[others[int(rng.integers(len(others)))]].particle.pattern).elements())
        if donated:
            dropped = next(iter(target))
            target[dropped] -= 1
            if target[dropped] == 0:
                del target[dropped]
            target[donated[int(rng.integers(len(donated)))]] += 1

    return target


def _moves_toward(current: Particle, target: Counter) -> list[Move]:
    """The edits that take ``current`` to the held exemplar."""
    cur = _template_counter(current.pattern)
    moves: list[Move] = []
    for template, count in target.items():
        surplus = count - cur.get(template, 0)
        moves.extend([Grow(template)] * surplus if surplus > 0
                     else [Shrink(template)] * (-surplus))
    for template, count in cur.items():
        if template not in target:
            moves.extend([Shrink(template)] * count)
    return moves


def _comprehensive_moves(current: Particle, pbests: list[Candidate], index: int,
                         pc: float, rng: np.random.Generator) -> list[Move]:
    """Assemble an exemplar and read the moves toward it in one step (used by the tests)."""
    return _moves_toward(current, _exemplar(pbests, index, pc, rng))


def optimize_clpso(hosts: list[Graph], stats: DatasetStats, params: Params,
                   seed: int, max_evaluations: int | None = None) -> Result:
    """CLPSO search. Same signature and Result as :func:`optimizer.optimize`."""
    rng = np.random.default_rng(seed)
    frequent_list = stats.frequent_edge_templates(params.min_support)
    frequent = set(frequent_list)
    if not frequent_list:
        return Result(top_k=[], gbest=None)

    pc = learning_probabilities(params.swarm_size)
    archive: dict[str, Candidate] = {}
    evaluator = make_evaluator(hosts, stats, params, max_evaluations)

    # Draw the starting templates in a seed-dependent order; taking them sorted gives every
    # run the same initial population (see pso.optimizer).
    order = list(rng.permutation(len(frequent_list)))
    frequent_list = [frequent_list[i] for i in order]


    def remember(candidate: Candidate) -> None:
        if candidate.evaluation.support < params.min_support:
            return
        existing = archive.get(candidate.signature)
        if existing is None or candidate.fitness > existing.fitness:
            archive[candidate.signature] = candidate

    currents: list[Candidate] = []
    pbests: list[Candidate] = []
    velocities: list[list[Move]] = []
    exemplars: list[list[Move] | None] = []
    stale: list[int] = []
    gbest: Candidate | None = None
    for i in range(params.swarm_size):
        template = frequent_list[i % len(frequent_list)]
        candidate = evaluator.score(_seed_particle(template, stats, params, rng))
        remember(candidate)
        currents.append(candidate)
        pbests.append(candidate)
        velocities.append([])
        exemplars.append(None)
        stale.append(0)
        if gbest is None or candidate.fitness > gbest.fitness:
            gbest = candidate

    # Iterate on the shared budget of distinct patterns, not a fixed round count.
    stalled = 0
    barren_restarts = 0
    replace_count = restart_count(params)
    while not evaluator.exhausted and barren_restarts < params.max_barren_restarts:
        before = evaluator.spent
        for i in range(params.swarm_size):
            if evaluator.exhausted:
                break
            # The exemplar is held until this particle's best has stalled, then rebuilt.
            if exemplars[i] is None or stale[i] >= params.clpso_refresh_gap:
                exemplars[i] = _exemplar(pbests, i, pc[i], rng)
                stale[i] = 0
            moves = _moves_toward(currents[i].particle, exemplars[i])
            velocity = compose(scale(velocities[i], params.omega, rng),
                               scale(moves, params.c1 * rng.random(), rng),
                               [], params.max_velocity)
            velocities[i] = velocity
            moved = apply_velocity(currents[i].particle, velocity, frequent, params)
            # Anti-stagnation: a velocity that leaves the particle in place jumps it to a
            # random valid neighbour, matching the basic swarm's behaviour.
            if canonical_signature(moved.pattern) == currents[i].signature:
                neighbours = _neighbors(currents[i].particle, frequent, params)
                if neighbours:
                    moved = neighbours[int(rng.integers(len(neighbours)))]
            # The pivot stays as the velocity left it (see pso.optimizer).
            if not is_valid(moved, frequent, params):
                continue
            candidate = evaluator.score(moved)
            remember(candidate)
            currents[i] = candidate
            if candidate.fitness > pbests[i].fitness:
                pbests[i] = candidate
                stale[i] = 0
            else:
                stale[i] += 1
            if candidate.fitness > gbest.fitness:
                gbest = candidate
        if evaluator.spent > before:
            stalled = 0
            barren_restarts = 0
            continue
        stalled += 1
        if stalled < STALL_ROUNDS:
            continue
        # Restart the weakest particles; their held exemplars go with them.
        stalled = 0
        barren_restarts += 1
        weakest = sorted(range(params.swarm_size),
                         key=lambda i: currents[i].fitness)[:replace_count]
        for i, particle in zip(weakest, fresh_particles(
                replace_count, frequent_list, stats, params, rng)):
            candidate = evaluator.score(particle)
            remember(candidate)
            currents[i] = candidate
            pbests[i] = candidate
            velocities[i] = []
            exemplars[i] = None
            stale[i] = 0

    pool = list(archive.values())
    # The attractor `gbest` may be an infrequent pattern; what a run reports may not.
    return Result(top_k=select_top_k(pool, params), gbest=reported_best(pool),
                  pool=pool)
