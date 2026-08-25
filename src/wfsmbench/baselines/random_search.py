"""Random-search baseline.

Given the same evaluation budget as the swarm, it samples patterns by taking a random
frequent seed edge and applying a random number of random valid extensions, then keeps
the Top-K by fitness. This is the standard sanity check for a metaheuristic: if a
structured search cannot beat blind sampling under an equal budget, it is not earning its
complexity.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..preprocess.statistics import DatasetStats
from ..pso.optimizer import Result, make_evaluator
from ._common import Archive, seed_particles, valid_children


def mine_random(hosts: list[Graph], stats: DatasetStats, params: Params,
                seed: int, max_evaluations: int) -> Result:
    rng = np.random.default_rng(seed)
    frequent = stats.frequent_edge_templates(params.min_support)
    archive = Archive(params)
    if not frequent:
        return archive.result()

    seeds = seed_particles(frequent)
    frequent_set = set(frequent)
    evaluator = make_evaluator(hosts, stats, params, max_evaluations)
    # Budgets count distinct patterns, so a draw that repeats one already scored is free and
    # simply does not advance the budget; the attempt cap stops an exhausted space spinning.
    attempts = 0
    while not evaluator.exhausted and attempts < max_evaluations * 20:
        attempts += 1
        particle = seeds[int(rng.integers(len(seeds)))]
        steps = int(rng.integers(0, params.max_edges))   # 0..max_edges-1 random extensions
        for _ in range(steps):
            children = valid_children(particle, frequent_set, params)
            if not children:
                break
            particle = children[int(rng.integers(len(children)))]
        archive.add(evaluator.score(particle))
    return archive.result()
