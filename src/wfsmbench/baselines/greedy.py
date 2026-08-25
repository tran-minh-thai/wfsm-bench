"""Greedy beam-search baseline.

A deterministic structured search: start from the frequent edges, and at each level keep
only the ``beam_width`` highest-fitness patterns, expanding those to the next size. It is
a middle ground between blind sampling and full enumeration — if the swarm cannot beat a
simple greedy beam, its population dynamics add little.
"""

from __future__ import annotations

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..core.signature import canonical_signature
from ..preprocess.statistics import DatasetStats
from ..pso.optimizer import Result
from ._common import Archive, evaluate, seed_particles, valid_children


def beam_search(hosts: list[Graph], stats: DatasetStats, params: Params,
                beam_width: int,
                max_levels: int | None = None,
                max_seeds: int | None = None,
                max_evaluations: int | None = None) -> tuple[Archive, list[Particle], int]:
    """Run the beam and return what it scored, the surviving frontier, and the number of
    pattern evaluations it spent.

    ``max_levels`` bounds the number of expansion rounds, ``max_seeds`` bounds how many
    frequent edge templates are scored at level 0, and ``max_evaluations`` stops the search
    once that many patterns have been evaluated. The evaluation cap is what makes the beam
    budget-comparable to the swarm: run to exhaustion it is near-exhaustive on a small
    space, so a fair head-to-head gives it the same evaluation allowance the swarm has.
    """
    frequent = stats.frequent_edge_templates(params.min_support)
    frequent_set = set(frequent)      # the full set still constrains expansion
    archive = Archive(params)
    seen: set[str] = set()

    def budget_left() -> bool:
        return max_evaluations is None or len(seen) < max_evaluations

    seeds = frequent if max_seeds is None else frequent[:max_seeds]
    beam: list[tuple[float, Particle]] = []
    for particle in seed_particles(seeds):
        if not budget_left():
            break
        seen.add(canonical_signature(particle.pattern))
        candidate = evaluate(particle, hosts, stats, params)
        archive.add(candidate)
        if candidate.evaluation.support >= params.min_support:
            beam.append((candidate.fitness, particle))
    beam.sort(key=lambda item: item[0], reverse=True)
    beam = beam[:beam_width]

    levels = 0
    while beam and budget_left() and (max_levels is None or levels < max_levels):
        levels += 1
        scored: list[tuple[float, Particle]] = []
        for _fitness, particle in beam:
            for child in valid_children(particle, frequent_set, params):
                if not budget_left():
                    break
                signature = canonical_signature(child.pattern)
                if signature in seen:
                    continue
                seen.add(signature)
                candidate = evaluate(child, hosts, stats, params)
                archive.add(candidate)
                if candidate.evaluation.support >= params.min_support:
                    scored.append((candidate.fitness, child))
            if not budget_left():
                break
        scored.sort(key=lambda item: item[0], reverse=True)
        beam = scored[:beam_width]

    return archive, [particle for _fitness, particle in beam], len(seen)


def mine_greedy(hosts: list[Graph], stats: DatasetStats, params: Params,
                beam_width: int, max_evaluations: int | None = None) -> Result:
    """Greedy beam search. With ``max_evaluations`` the beam is budget-capped for a fair
    head-to-head with the fixed-budget searches; without it, it runs to exhaustion."""
    archive, _frontier, _evaluations = beam_search(
        hosts, stats, params, beam_width, max_evaluations=max_evaluations)
    return archive.result()
