"""Genetic-algorithm baseline.

An elitist GA over the same pattern space and objective: tournament selection, a
template-transfer crossover (grow one parent by an edge template drawn from the other),
and grow/shrink mutation. Population size and generation count reuse the swarm size and
iteration count, so the GA and the PSO run at a matched search budget. This answers the
natural "why particle swarm rather than another population metaheuristic?" question.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..preprocess.statistics import DatasetStats
from ..preprocess.template import EdgeTemplate
from ..pso.edits import Grow, Shrink, apply_move, is_valid
from ..pso.optimizer import STALL_ROUNDS, Result, make_evaluator, restart_count
from ._common import (
    Archive,
    as_particle,
    evaluate,
    pattern_templates,
    seed_particles,
    valid_children,
)


def _tournament(scored: list[tuple[float, Particle]], rng: np.random.Generator) -> Particle:
    i, j = rng.integers(0, len(scored), size=2)
    return scored[i][1] if scored[i][0] >= scored[j][0] else scored[j][1]


def _recombine(parent_a: Particle, parent_b: Particle,
               frequent_set: set[EdgeTemplate], params: Params,
               rng: np.random.Generator) -> Particle:
    templates = pattern_templates(parent_b.pattern)
    if templates:
        template = templates[int(rng.integers(len(templates)))]
        child = apply_move(parent_a, Grow(template))
        if child is not None:
            candidate = as_particle(child.pattern)
            if is_valid(candidate, frequent_set, params):
                return candidate
    return parent_a


def _mutate(particle: Particle, frequent_set: set[EdgeTemplate], params: Params,
            rng: np.random.Generator) -> Particle:
    if rng.random() < 0.5:
        children = valid_children(particle, frequent_set, params)
        if children:
            return children[int(rng.integers(len(children)))]
    else:
        templates = pattern_templates(particle.pattern)
        if templates:
            shrunk = apply_move(particle, Shrink(templates[int(rng.integers(len(templates)))]))
            if shrunk is not None and is_valid(shrunk, frequent_set, params):
                return shrunk
    return particle


def mine_genetic(hosts: list[Graph], stats: DatasetStats, params: Params, seed: int,
                 max_evaluations: int | None = None) -> Result:
    rng = np.random.default_rng(seed)
    frequent = stats.frequent_edge_templates(params.min_support)
    frequent_set = set(frequent)
    archive = Archive(params)
    if not frequent:
        return archive.result()

    seeds = seed_particles(frequent)
    evaluator = make_evaluator(hosts, stats, params, max_evaluations)

    def score(particle: Particle) -> float:
        candidate = evaluator.score(particle)
        archive.add(candidate)
        return candidate.fitness

    population = [seeds[int(rng.integers(len(seeds)))] for _ in range(params.swarm_size)]
    scored = [(score(p), p) for p in population]

    # Run generations until the shared budget is spent; a generation that scores nothing new
    # means the population has converged onto patterns already paid for.
    stalled = 0
    barren_restarts = 0
    replace_count = restart_count(params)
    while not evaluator.exhausted and barren_restarts < params.max_barren_restarts:
        before = evaluator.spent
        scored.sort(key=lambda item: item[0], reverse=True)
        elite_count = max(1, int(params.swarm_size * params.elite_fraction))
        elite = scored[:elite_count]

        offspring: list[Particle] = []
        while len(offspring) < params.swarm_size - elite_count:
            parent_a = _tournament(scored, rng)
            parent_b = _tournament(scored, rng)
            child = _recombine(parent_a, parent_b, frequent_set, params, rng)
            child = _mutate(child, frequent_set, params, rng)
            offspring.append(child)

        population = [p for _fitness, p in elite] + offspring
        scored = [(score(p), p) for p in population]
        if evaluator.spent > before:
            stalled = 0
            barren_restarts = 0
            continue
        stalled += 1
        if stalled < STALL_ROUNDS:
            continue
        # A converged population is replaced by immigrants, keeping only its best.
        stalled = 0
        barren_restarts += 1
        scored.sort(key=lambda item: item[0], reverse=True)
        keep = scored[:max(1, len(scored) - replace_count)]
        immigrants = [seeds[int(rng.integers(len(seeds)))] for _ in range(replace_count)]
        scored = keep + [(score(p), p) for p in immigrants]

    return archive.result()
