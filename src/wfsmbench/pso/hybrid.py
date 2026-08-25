"""Beam-seeded particle swarm — a hybrid of the two strongest search strategies.

The comparison study showed the two methods fail in opposite ways: the deterministic beam
reaches high fitness but returns near-duplicates of one structure, while the swarm explores
widely but spends much of its budget climbing from single-edge seeds. The hybrid buys a few
strong patterns with a narrow beam and hands them to the swarm as starting positions.

Budget stays comparable to a plain run: the evaluations the warm start consumes are turned
into swarm-iteration equivalents and deducted from the swarm's own iterations, so
``mine_hybrid`` and ``optimize`` are given the same total evaluation allowance. That makes
the warm start genuinely costly — a beam level scores every child of every beam member,
which buys one level of depth for several iterations' worth of budget — so the beam is kept
deliberately narrow and seeded from only as many templates as the swarm itself would score.
"""

from __future__ import annotations

from dataclasses import replace

from ..baselines.greedy import beam_search
from ..config import Params
from ..core.graph import Graph
from ..preprocess.statistics import DatasetStats
from .optimizer import Result, optimize


def mine_hybrid(hosts: list[Graph], stats: DatasetStats, params: Params, seed: int,
                beam_width: int | None = None,
                warmstart_levels: int | None = None) -> Result:
    width = beam_width if beam_width is not None else max(2, params.swarm_size // 5)
    archive, frontier, evaluations = beam_search(
        hosts, stats, params, width,
        max_levels=warmstart_levels, max_seeds=params.swarm_size)

    known = archive.candidates()
    if not known:
        return optimize(hosts, stats, params, seed)

    # A beam run to exhaustion ends with an empty frontier; fall back to the best patterns
    # it found so the warm start is never wasted.
    initial = frontier or [c.particle for c in
                           sorted(known, key=lambda c: c.fitness, reverse=True)[:params.swarm_size]]

    spent_iterations = -(-evaluations // max(1, params.swarm_size))   # ceiling division
    remaining = max(0, params.iterations - spent_iterations)
    return optimize(hosts, stats, replace(params, iterations=remaining), seed,
                    initial=initial, known=known)
