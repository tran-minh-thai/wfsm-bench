"""Iterated local search — the single-solution baseline the comparison was missing.

Every other method here keeps a population or a beam. Nothing kept just one solution and
walked uphill, which left the most basic question unasked: does any of the population
machinery earn its complexity?

The measurement that makes the question urgent is the local-optimum count. On Cuneiform
(n=60, size 6, min_support 2) the corrected objective has **22** local optima over a pool of
42523 patterns, against 3233 under the uncorrected one -- and most of that 3233 were patterns
sitting at the size bound with nowhere to grow, an artefact of the bound rather than
ruggedness. A landscape with 22 local optima is where a hill-climber with restarts is
supposed to win, so it has to be given the chance.

Three decisions, each one a place the baseline could be made artificially weak:

*First improvement, not steepest ascent.* Budgets here count distinct patterns scored, and
steepest ascent pays the whole neighbourhood -- tens of evaluations -- for one step, while
first improvement pays until it finds an improving move. On a near-unimodal landscape both
end at the same optimum, so charging the baseline for the expensive version would understate
it. Neighbours are visited in a seed-dependent order so the walk is not deterministic.

*Perturbation of at least two moves.* One random edit is undone by one hill-climbing step,
which turns the search into a random walk over the same basin. Two or more is what lets the
kick reach a different basin.

*Better-acceptance, then restart.* A new local optimum is adopted only if it beats the one
the kick came from, which is standard ILS. When several kicks in a row find nothing new the
walk restarts from a fresh seed, matching the stall rule every other method here follows --
without it a method quietly refuses a larger budget.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..preprocess.statistics import DatasetStats
from ..preprocess.template import EdgeTemplate
from ..pso.edits import is_valid
from ..pso.optimizer import STALL_ROUNDS, Result, _neighbors, make_evaluator
from ._common import Archive, seed_particles

# How many random edits a kick applies. One is undone by a single hill-climbing step.
PERTURBATION = 2


def _climb(particle: Particle, fitness: float, frequent: set[EdgeTemplate], params: Params,
           evaluator, archive: Archive, rng: np.random.Generator) -> tuple[Particle, float]:
    """Walk uphill by first improvement until no neighbour is fitter, or the budget ends."""
    current, current_fitness = particle, fitness
    while not evaluator.exhausted:
        neighbours = _neighbors(current, frequent, params)
        if not neighbours:
            break
        improved = False
        for index in rng.permutation(len(neighbours)):
            if evaluator.exhausted:
                break
            candidate = evaluator.score(neighbours[int(index)])
            archive.add(candidate)
            if candidate.fitness > current_fitness:
                current, current_fitness = candidate.particle, candidate.fitness
                improved = True
                break
        if not improved:
            break
    return current, current_fitness


def _kick(particle: Particle, frequent: set[EdgeTemplate], params: Params,
          rng: np.random.Generator) -> Particle:
    """Several random valid edits, to land in a different basin than the one just climbed."""
    current = particle
    for _ in range(PERTURBATION):
        neighbours = _neighbors(current, frequent, params)
        if not neighbours:
            break
        current = neighbours[int(rng.integers(len(neighbours)))]
    return current


def mine_local_search(hosts: list[Graph], stats: DatasetStats, params: Params, seed: int,
                      max_evaluations: int | None = None) -> Result:
    rng = np.random.default_rng(seed)
    frequent_list = stats.frequent_edge_templates(params.min_support)
    frequent = set(frequent_list)
    archive = Archive(params)
    if not frequent_list:
        return archive.result()

    seeds = seed_particles(frequent_list)
    evaluator = make_evaluator(hosts, stats, params, max_evaluations)

    def start() -> tuple[Particle, float]:
        particle = seeds[int(rng.integers(len(seeds)))]
        candidate = evaluator.score(particle)
        archive.add(candidate)
        return candidate.particle, candidate.fitness

    incumbent, incumbent_fitness = start()
    incumbent, incumbent_fitness = _climb(
        incumbent, incumbent_fitness, frequent, params, evaluator, archive, rng)

    stalled = 0
    barren_restarts = 0
    while not evaluator.exhausted and barren_restarts < params.max_barren_restarts:
        before = evaluator.spent
        kicked = _kick(incumbent, frequent, params, rng)
        if not is_valid(kicked, frequent, params):
            kicked = incumbent
        candidate = evaluator.score(kicked)
        archive.add(candidate)
        found, found_fitness = _climb(candidate.particle, candidate.fitness,
                                      frequent, params, evaluator, archive, rng)
        if found_fitness > incumbent_fitness:
            incumbent, incumbent_fitness = found, found_fitness

        if evaluator.spent > before:
            stalled = 0
            barren_restarts = 0
            continue
        stalled += 1
        if stalled < STALL_ROUNDS:
            continue
        # Kicking from here keeps landing on patterns already paid for: start over.
        stalled = 0
        barren_restarts += 1
        fresh, fresh_fitness = start()
        fresh, fresh_fitness = _climb(fresh, fresh_fitness, frequent, params,
                                      evaluator, archive, rng)
        if fresh_fitness > incumbent_fitness:
            incumbent, incumbent_fitness = fresh, fresh_fitness

    return archive.result()
