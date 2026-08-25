"""Multiprocess execution of method/seed jobs.

Subgraph matching is CPU-bound, so the independent (method, seed) runs are spread across
processes (threads would not help under the GIL). The shared inputs — the graph subset,
statistics, and parameters — are sent to each worker once via an initializer, not once per
job, so large datasets are pickled ``workers`` times rather than ``jobs`` times. Results are
deterministic regardless of the worker count, since every run is independently seeded.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from ..baselines import mine_aco, mine_exhaustive, mine_genetic, mine_greedy, mine_random
from ..measure import mean_pairwise_diversity
from ..measure.pareto import objectives, pareto_front
from ..pso import optimize
from ..pso.hybrid import mine_hybrid
from ..pso.clpso import optimize_clpso
from ..pso.multiswarm import optimize_multiswarm
from ..pso.optimizer import Result, select_top_k

_STATE: dict = {}


def resolve_workers(workers: int | None) -> int:
    if workers and workers > 0:
        return workers
    return max(1, (os.cpu_count() or 1) - 1)


def _init(subset, stats, params, budget, weights=None) -> None:
    _STATE.update(subset=subset, stats=stats, params=params, budget=budget,
                  weights=weights or [])


def _search(method: str, seed: int) -> tuple[Result, float]:
    """Run one method and time it. Shared by every job flavour so they cannot drift."""
    subset, stats, params, budget = (
        _STATE["subset"], _STATE["stats"], _STATE["params"], _STATE["budget"])
    start = time.perf_counter()
    if method == "exhaustive":
        result = mine_exhaustive(subset, stats, params)
    elif method == "greedy":
        result = mine_greedy(subset, stats, params, params.swarm_size)
    elif method == "greedy_budget":
        # Beam capped to the same evaluation allowance as the fixed-budget searches, for a
        # fair head-to-head; run-to-exhaustion greedy is kept as the separate "greedy".
        result = mine_greedy(subset, stats, params, params.swarm_size,
                             max_evaluations=budget)
    elif method == "pso":
        result = optimize(subset, stats, params, seed)
    elif method == "random":
        result = mine_random(subset, stats, params, seed, budget)
    elif method == "genetic":
        result = mine_genetic(subset, stats, params, seed)
    elif method == "aco":
        result = mine_aco(subset, stats, params, seed, max_evaluations=budget)
    elif method == "hybrid":
        result = mine_hybrid(subset, stats, params, seed)
    elif method == "clpso":
        result = optimize_clpso(subset, stats, params, seed)
    elif method == "multiswarm":
        result = optimize_multiswarm(subset, stats, params, seed)
    else:
        raise ValueError(f"unknown method: {method}")
    return result, time.perf_counter() - start


def _run(job: tuple[str, int]) -> tuple[str, int, float, float, float]:
    method, seed = job
    result, runtime = _search(method, seed)
    fitness = result.gbest.fitness if result.gbest else 0.0
    diversity = mean_pairwise_diversity([c.particle.pattern for c in result.top_k])
    return method, seed, fitness, runtime, diversity


def _run_weights(job: tuple[str, int]) -> tuple[str, int, float, float, list[float]]:
    """One search run, reported under every Top-K admission weight in ``_STATE['weights']``.

    The admission weight does not affect the search, so re-running the same seed per weight
    would only repeat identical work. Scoring one run's pool at each weight is both cheaper
    and makes the fitness column identical across weights by construction.
    """
    method, seed = job
    params = _STATE["params"]
    result, runtime = _search(method, seed)
    diversities = []
    for weight in _STATE["weights"]:
        top = select_top_k(result.pool, replace(params, diversity_weight=weight))
        diversities.append(mean_pairwise_diversity([c.particle.pattern for c in top]))
    return method, seed, (result.gbest.fitness if result.gbest else 0.0), runtime, diversities


def run_weight_jobs(jobs: list[tuple[str, int]], subset, stats, params, budget,
                    weights: list[float], workers: int | None):
    workers = resolve_workers(workers)
    initargs = (subset, stats, params, budget, weights)
    if workers <= 1 or len(jobs) <= 1:
        _init(*initargs)
        return [_run_weights(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=initargs) as pool:
        return list(pool.map(_run_weights, jobs))


def _run_front(job: tuple[str, int]):
    """One search run, reduced to the Pareto front of its whole candidate pool.

    The pool rather than the Top-K, because the Top-K is selected by the scalar fitness and
    would hide trade-off patterns the run actually found.
    """
    method, seed = job
    result, runtime = _search(method, seed)
    points = [objectives(c.evaluation, c.particle.pattern) for c in result.pool]
    front = pareto_front(points) if points else []
    return method, seed, front, (result.gbest.fitness if result.gbest else 0.0), runtime


def run_front_jobs(jobs: list[tuple[str, int]], subset, stats, params, budget,
                   workers: int | None):
    workers = resolve_workers(workers)
    initargs = (subset, stats, params, budget)
    if workers <= 1 or len(jobs) <= 1:
        _init(*initargs)
        return [_run_front(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=initargs) as pool:
        return list(pool.map(_run_front, jobs))


def run_jobs(jobs: list[tuple[str, int]], subset, stats, params, budget,
             workers: int | None) -> list[tuple[str, int, float, float, float]]:
    workers = resolve_workers(workers)
    if workers <= 1 or len(jobs) <= 1:
        _init(subset, stats, params, budget)
        return [_run(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(subset, stats, params, budget)) as pool:
        return list(pool.map(_run, jobs))
