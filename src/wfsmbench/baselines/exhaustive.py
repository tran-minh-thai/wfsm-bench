"""Exhaustive enumeration baseline — the ground-truth Top-K by brute force.

Enumerates every valid canonical pattern up to the configured size bounds and scores
each with the SAME measure and fitness the PSO search uses, then returns the exact
Top-K. This is the fair reference for the swarm: identical objective F(g), but complete
and deterministic instead of stochastic, so it establishes the true optimum the PSO is
trying to reach.

Enumeration is a breadth-first expansion from the frequent edge templates, adding one
frequent-template edge at a time, deduplicated by canonical signature. Support is
anti-monotone, so a pattern is expanded only when it is itself frequent — this keeps the
enumeration finite and focused while still reaching every frequent pattern.
"""

from __future__ import annotations

from typing import Callable

from ..config import Params
from ..core.graph import Graph
from ..core.signature import canonical_signature
from ..preprocess.statistics import DatasetStats
from ..pso.edits import is_valid
from ..pso.optimizer import Result
from ._common import Archive, as_particle, evaluate, extensions, seed_particles


def mine_exhaustive(hosts: list[Graph], stats: DatasetStats, params: Params,
                    progress: Callable[[int, int], None] | None = None) -> Result:
    """Every frequent pattern within the size bounds, scored, with the exact Top-K.

    ``progress``, when given, is called after each pattern is expanded with the number expanded
    so far and the number of frequent patterns found so far. It only observes: what is
    enumerated and in which order does not depend on it.
    """
    frequent = set(stats.frequent_edge_templates(params.min_support))
    archive = Archive(params)
    seen: set[str] = set()
    queue: list[Graph] = []

    def consider(pattern: Graph) -> None:
        signature = canonical_signature(pattern)
        if signature in seen:
            return
        seen.add(signature)
        particle = as_particle(pattern)
        if not is_valid(particle, frequent, params):   # enforces the size bounds
            return
        candidate = evaluate(particle, hosts, stats, params)
        if candidate.evaluation.support >= params.min_support:
            archive.add(candidate)
            queue.append(pattern)   # expand only frequent patterns (anti-monotone support)

    for particle in seed_particles(sorted(frequent)):
        consider(particle.pattern)

    index = 0
    while index < len(queue):
        pattern = queue[index]
        index += 1
        for child in extensions(pattern, frequent):
            consider(child)
        if progress is not None:
            progress(index, len(queue))

    return archive.result()
