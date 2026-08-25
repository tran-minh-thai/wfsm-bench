"""Dynamic multi-swarm PSO (DMS-PSO) over the discrete graph-edit space.

Where the basic swarm has one global best that every particle is pulled toward -- the reason
it converges early -- a multi-swarm splits the population into small sub-swarms, each with
its own local best. Particles therefore exploit several regions at once instead of one, and
periodically the sub-swarms are randomly regrouped so information still spreads
(Liang & Suganthan, 2005).

This differs from the comprehensive-learning variant in an important way. CLPSO weakens
exploitation everywhere by learning each dimension from a different exemplar; a multi-swarm
keeps exploitation strong *inside* each sub-swarm and gets its diversity from having several
of them. Two consequences are worth measuring here: k independent sub-swarms behave like a
portfolio of shorter runs, whose best can beat one long run on an easy landscape; and
distinct sub-swarms may settle on distinct trade-offs, which shows up as Pareto coverage
rather than as a better single score.

The objective, archive, validity rules, and Top-K admission are shared with
:mod:`wfsmbench.pso.optimizer`, so only the social topology differs.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..core.signature import canonical_signature
from ..preprocess.statistics import DatasetStats
from .edits import Move, apply_velocity, diff, is_valid
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


def partition(size: int, groups: int, rng: np.random.Generator) -> list[list[int]]:
    """Randomly split ``range(size)`` into ``groups`` nearly equal sub-swarms."""
    order = list(rng.permutation(size))
    groups = max(1, min(groups, size))
    return [order[i::groups] for i in range(groups)]


def optimize_multiswarm(hosts: list[Graph], stats: DatasetStats, params: Params,
                        seed: int, max_evaluations: int | None = None) -> Result:
    """DMS-PSO search. Same signature and Result as :func:`optimizer.optimize`.

    Uses ``params.num_subswarms`` sub-swarms, regrouped every ``params.regroup_period``
    iterations. One sub-swarm reproduces the basic swarm's star topology.
    """
    rng = np.random.default_rng(seed)
    frequent_list = stats.frequent_edge_templates(params.min_support)
    frequent = set(frequent_list)
    if not frequent_list:
        return Result(top_k=[], gbest=None)

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
    gbest: Candidate | None = None
    for i in range(params.swarm_size):
        template = frequent_list[i % len(frequent_list)]
        candidate = evaluator.score(_seed_particle(template, stats, params, rng))
        remember(candidate)
        currents.append(candidate)
        pbests.append(candidate)
        velocities.append([])
        if gbest is None or candidate.fitness > gbest.fitness:
            gbest = candidate

    groups = partition(params.swarm_size, params.num_subswarms, rng)

    iteration = 0
    # Iterate on the shared budget of distinct patterns, not a fixed round count.
    stalled = 0
    barren_restarts = 0
    replace_count = restart_count(params)
    while not evaluator.exhausted and barren_restarts < params.max_barren_restarts:
        before = evaluator.spent
        iteration += 1
        if iteration and iteration % params.regroup_period == 0:
            groups = partition(params.swarm_size, params.num_subswarms, rng)

        for group in groups:
            # The social attractor is the best personal best inside this sub-swarm only.
            leader = max((pbests[i] for i in group), key=lambda c: c.fitness)
            for i in group:
                velocity = compose(
                    scale(velocities[i], params.omega, rng),
                    scale(diff(currents[i].particle, pbests[i].particle),
                          params.c1 * rng.random(), rng),
                    scale(diff(currents[i].particle, leader.particle),
                          params.c2 * rng.random(), rng),
                    params.max_velocity)
                velocities[i] = velocity
                moved = apply_velocity(currents[i].particle, velocity, frequent, params)
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
                if candidate.fitness > gbest.fitness:
                    gbest = candidate
        if evaluator.spent > before:
            stalled = 0
            barren_restarts = 0
            continue
        stalled += 1
        if stalled < STALL_ROUNDS:
            continue
        # Restart the weakest particles and regroup, so the new members mix in.
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
        groups = partition(params.swarm_size, params.num_subswarms, rng)

    pool = list(archive.values())
    # The attractor `gbest` may be an infrequent pattern; what a run reports may not.
    return Result(top_k=select_top_k(pool, params), gbest=reported_best(pool),
                  pool=pool)
