"""Experiment harness (proposal section 5): run the search over a graph subset,
repeat across seeds, and report mean/std of the metrics.

A run selects the first ``graph_limit`` graphs, builds statistics on that subset, and
runs the PSO once per seed. Metrics are the best-pattern outcome per run — runtime,
fitness, support, WSUP, ATW, and pattern size — aggregated across seeds. Varying
``graph_limit`` gives the scalability curve; varying ``Params.pivot_strategy`` gives the
pivot ablation.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..baselines import mine_exhaustive, mine_genetic, mine_greedy, mine_random
from ..config import Params
from ..measure import mean_pairwise_diversity
from ..measure.pareto import coverage, hypervolume_ratio, objectives, pareto_front
from ..core.graph import Graph
from ..preprocess.statistics import build_statistics
from ..pso import optimize
from .parallel import run_front_jobs, run_jobs, run_weight_jobs


@dataclass(frozen=True)
class RunMetrics:
    runtime: float
    fitness: float
    support: int
    wsup: float
    atw: float
    pattern_nodes: int
    pattern_edges: int


@dataclass(frozen=True)
class Aggregate:
    dataset: str
    graph_limit: int
    pivot_strategy: str
    num_seeds: int
    runtime_mean: float
    runtime_std: float
    fitness_mean: float
    fitness_std: float
    support_mean: float
    wsup_mean: float
    atw_mean: float
    mode_nodes: int
    mode_edges: int


def run_once(subset: list[Graph], stats, params: Params, seed: int) -> RunMetrics:
    start = time.perf_counter()
    result = optimize(subset, stats, params, seed)
    runtime = time.perf_counter() - start
    best = result.gbest
    if best is None:
        return RunMetrics(runtime, 0.0, 0, 0.0, 0.0, 0, 0)
    return RunMetrics(
        runtime=runtime,
        fitness=best.fitness,
        support=best.evaluation.support,
        wsup=best.evaluation.wsup,
        atw=best.evaluation.atw,
        pattern_nodes=best.particle.pattern.num_nodes(),
        pattern_edges=best.particle.pattern.num_edges(),
    )


def run_experiment(graphs: list[Graph], dataset: str, graph_limit: int,
                   seeds: list[int], params: Params) -> Aggregate:
    subset = graphs[:graph_limit]
    stats = build_statistics(subset)
    metrics = [run_once(subset, stats, params, seed) for seed in seeds]

    sizes = Counter((m.pattern_nodes, m.pattern_edges) for m in metrics)
    (mode_nodes, mode_edges), _ = sizes.most_common(1)[0]

    def mean(attr):
        return float(np.mean([getattr(m, attr) for m in metrics]))

    def std(attr):
        return float(np.std([getattr(m, attr) for m in metrics]))

    return Aggregate(
        dataset=dataset,
        graph_limit=graph_limit,
        pivot_strategy=params.pivot_strategy,
        num_seeds=len(seeds),
        runtime_mean=mean("runtime"),
        runtime_std=std("runtime"),
        fitness_mean=mean("fitness"),
        fitness_std=std("fitness"),
        support_mean=mean("support"),
        wsup_mean=mean("wsup"),
        atw_mean=mean("atw"),
        mode_nodes=mode_nodes,
        mode_edges=mode_edges,
    )


@dataclass(frozen=True)
class Comparison:
    dataset: str
    graph_limit: int
    num_seeds: int
    num_frequent: int          # how many frequent patterns the exhaustive search found
    exhaustive_fitness: float  # the true optimum F
    exhaustive_runtime: float
    pso_fitness_mean: float
    pso_fitness_std: float
    pso_runtime_mean: float
    fitness_gap_pct: float     # (exhaustive - pso_mean) / exhaustive * 100
    speedup: float             # exhaustive_runtime / pso_runtime_mean
    pso_hits_optimum_rate: float  # fraction of seeds whose gbest is in the exhaustive Top-K


def run_comparison(graphs: list[Graph], dataset: str, graph_limit: int,
                   seeds: list[int], params: Params) -> Comparison:
    subset = graphs[:graph_limit]
    stats = build_statistics(subset)

    start = time.perf_counter()
    exhaustive = mine_exhaustive(subset, stats, params)
    exhaustive_runtime = time.perf_counter() - start
    exhaustive_fitness = exhaustive.gbest.fitness if exhaustive.gbest else 0.0
    optimum_signatures = {c.signature for c in exhaustive.top_k}

    fitnesses, runtimes, hits = [], [], 0
    for seed in seeds:
        start = time.perf_counter()
        result = optimize(subset, stats, params, seed)
        runtimes.append(time.perf_counter() - start)
        fitnesses.append(result.gbest.fitness if result.gbest else 0.0)
        if result.gbest and result.gbest.signature in optimum_signatures:
            hits += 1

    pso_fitness_mean = float(np.mean(fitnesses))
    pso_runtime_mean = float(np.mean(runtimes))
    gap = (exhaustive_fitness - pso_fitness_mean) / exhaustive_fitness * 100 \
        if exhaustive_fitness > 0 else 0.0
    speedup = exhaustive_runtime / pso_runtime_mean if pso_runtime_mean > 0 else 0.0

    return Comparison(
        dataset=dataset,
        graph_limit=graph_limit,
        num_seeds=len(seeds),
        num_frequent=len(exhaustive.top_k),
        exhaustive_fitness=exhaustive_fitness,
        exhaustive_runtime=exhaustive_runtime,
        pso_fitness_mean=pso_fitness_mean,
        pso_fitness_std=float(np.std(fitnesses)),
        pso_runtime_mean=pso_runtime_mean,
        fitness_gap_pct=gap,
        speedup=speedup,
        pso_hits_optimum_rate=hits / len(seeds),
    )


@dataclass(frozen=True)
class BudgetPoint:
    dataset: str
    graph_limit: int
    swarm_size: int
    iterations: int
    evaluations: int           # ~S*(1+T): the search effort proxy
    num_seeds: int
    exhaustive_fitness: float  # constant across the sweep (same search space)
    pso_fitness_mean: float
    pso_fitness_std: float
    pso_runtime_mean: float
    fitness_gap_pct: float
    pso_hits_optimum_rate: float


def run_budget_sweep(graphs: list[Graph], dataset: str, graph_limit: int,
                     budgets: list[tuple[int, int]], seeds: list[int],
                     base_params: Params) -> list[BudgetPoint]:
    """Vary the PSO budget (swarm size, iterations) against a fixed exhaustive optimum."""
    subset = graphs[:graph_limit]
    stats = build_statistics(subset)
    exhaustive = mine_exhaustive(subset, stats, base_params)
    exhaustive_fitness = exhaustive.gbest.fitness if exhaustive.gbest else 0.0
    optimum_signatures = {c.signature for c in exhaustive.top_k}

    points: list[BudgetPoint] = []
    for swarm_size, iterations in budgets:
        params = replace(base_params, swarm_size=swarm_size, iterations=iterations)
        fitnesses, runtimes, hits = [], [], 0
        for seed in seeds:
            start = time.perf_counter()
            result = optimize(subset, stats, params, seed)
            runtimes.append(time.perf_counter() - start)
            fitnesses.append(result.gbest.fitness if result.gbest else 0.0)
            if result.gbest and result.gbest.signature in optimum_signatures:
                hits += 1
        pso_mean = float(np.mean(fitnesses))
        gap = (exhaustive_fitness - pso_mean) / exhaustive_fitness * 100 \
            if exhaustive_fitness > 0 else 0.0
        points.append(BudgetPoint(
            dataset=dataset,
            graph_limit=graph_limit,
            swarm_size=swarm_size,
            iterations=iterations,
            evaluations=swarm_size * (1 + iterations),
            num_seeds=len(seeds),
            exhaustive_fitness=exhaustive_fitness,
            pso_fitness_mean=pso_mean,
            pso_fitness_std=float(np.std(fitnesses)),
            pso_runtime_mean=float(np.mean(runtimes)),
            fitness_gap_pct=gap,
            pso_hits_optimum_rate=hits / len(seeds),
        ))
    return points


@dataclass(frozen=True)
class MethodResult:
    dataset: str
    graph_limit: int
    method: str
    num_seeds: int
    fitness_mean: float
    fitness_std: float
    runtime_mean: float
    gap_pct: float
    max_size: int = 0            # max pattern size (nodes/edges) of the run; 0 = unset
    diversity_mean: float = 0.0  # mean pairwise Top-K structural diversity


def run_method_comparison(graphs: list[Graph], dataset: str, graph_limit: int,
                          seeds: list[int], params: Params,
                          include_exhaustive: bool = True,
                          include_greedy: bool = True,
                          workers: int = 1) -> list[MethodResult]:
    """Compare the search methods at a matched evaluation budget.

    ``include_exhaustive`` and ``include_greedy`` can be turned off for large search
    spaces where complete or beam search is intractable. The fitness gap is measured
    against the exhaustive optimum when it is present, otherwise against the best method.
    ``workers`` > 1 (or None for auto) spreads the (method, seed) runs across processes.
    """
    subset = graphs[:graph_limit]
    stats = build_statistics(subset)
    budget = params.swarm_size * (1 + params.iterations)   # matched PSO-equivalent evaluations

    jobs: list[tuple[str, int]] = []
    if include_exhaustive:
        jobs.append(("exhaustive", -1))
    if include_greedy:
        jobs.append(("greedy", -1))
    for method in ("pso", "random", "genetic", "hybrid"):
        jobs.extend((method, seed) for seed in seeds)

    outcomes = run_jobs(jobs, subset, stats, params, budget, workers)

    collected: dict[str, dict[str, list[float]]] = {}
    for method, _seed, fitness, runtime, diversity in outcomes:
        bucket = collected.setdefault(method, {"fitness": [], "runtime": [], "diversity": []})
        bucket["fitness"].append(fitness)
        bucket["runtime"].append(runtime)
        bucket["diversity"].append(diversity)

    means = {method: float(np.mean(vals["fitness"])) for method, vals in collected.items()}
    reference = means["exhaustive"] if "exhaustive" in means else (max(means.values()) if means else 0.0)

    ordered = [m for m in ("exhaustive", "greedy", "pso", "hybrid", "random", "genetic")
               if m in collected]
    rows: list[MethodResult] = []
    for method in ordered:
        vals = collected[method]
        mean = means[method]
        gap = (reference - mean) / reference * 100 if reference > 0 else 0.0
        rows.append(MethodResult(dataset, graph_limit, method, len(seeds),
                                 mean, float(np.std(vals["fitness"])), float(np.mean(vals["runtime"])),
                                 gap, params.max_nodes, float(np.mean(vals["diversity"]))))
    return rows


def run_size_sweep(graphs: list[Graph], dataset: str, graph_limit: int,
                   sizes: list[int], seeds: list[int], base_params: Params,
                   exhaustive_max_size: int = 3, workers: int = 1) -> list[MethodResult]:
    """Method comparison across a range of maximum pattern sizes.

    Exhaustive search is included only up to ``exhaustive_max_size`` (beyond which it is
    intractable on dense data); above that the gap is measured against the best method.

    Only the size bounds move: ``base_params.size_reference`` carries through unchanged, so
    every size is scored on one objective and the fitness column is comparable down it. Set
    that field on ``base_params`` to the largest size in ``sizes`` -- Params rejects an
    anchor below the bound, so a caller who forgets gets an error rather than a silently
    shifting objective.
    """
    rows: list[MethodResult] = []
    for size in sizes:
        params = replace(base_params, max_nodes=size, max_edges=size)
        rows.extend(run_method_comparison(
            graphs, dataset, graph_limit, seeds, params,
            include_exhaustive=size <= exhaustive_max_size,
            include_greedy=True, workers=workers))
    return rows


def method_comparison_to_csv(rows: list[MethodResult]) -> str:
    return _rows_to_csv(list(MethodResult.__annotations__.keys()), rows)


def _rows_to_csv(columns: list[str], rows: list) -> str:
    lines = [",".join(columns)]
    for row in rows:
        data = asdict(row)
        cells = [f"{data[c]:.6f}" if isinstance(data[c], float) else str(data[c])
                 for c in columns]
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def to_csv(aggregates: list[Aggregate]) -> str:
    return _rows_to_csv(list(Aggregate.__annotations__.keys()), aggregates)


def comparison_to_csv(comparisons: list[Comparison]) -> str:
    return _rows_to_csv(list(Comparison.__annotations__.keys()), comparisons)


def budget_to_csv(points: list[BudgetPoint]) -> str:
    return _rows_to_csv(list(BudgetPoint.__annotations__.keys()), points)


@dataclass(frozen=True)
class DiversityPoint:
    """One (method, admission weight) cell of the Top-K diversity sweep."""
    dataset: str
    graph_limit: int
    max_size: int
    method: str
    num_seeds: int
    diversity_weight: float
    fitness_mean: float
    diversity_mean: float
    diversity_std: float
    runtime_mean: float


def run_diversity_sweep(graphs: list[Graph], dataset: str, graph_limit: int,
                        seeds: list[int], params: Params, weights: list[float],
                        include_greedy: bool = True,
                        workers: int = 1) -> list[DiversityPoint]:
    """Report every method's Top-K under a range of admission weights.

    The weight changes only which patterns are reported, never the search, so each
    (method, seed) is searched ONCE and its candidate pool is re-scored at every weight.
    A consequence worth checking in the output: ``fitness_mean`` is identical across
    weights for a given method, because the best pattern found cannot depend on how the
    result set is later selected.
    """
    subset = graphs[:graph_limit]
    stats = build_statistics(subset)
    budget = params.swarm_size * (1 + params.iterations)

    jobs: list[tuple[str, int]] = []
    if include_greedy:
        jobs.append(("greedy", -1))
    for method in ("pso", "hybrid", "random", "genetic"):
        jobs.extend((method, seed) for seed in seeds)

    outcomes = run_weight_jobs(jobs, subset, stats, params, budget, weights, workers)

    collected: dict[str, dict] = {}
    for method, _seed, fitness, runtime, diversities in outcomes:
        bucket = collected.setdefault(
            method, {"fitness": [], "runtime": [], "diversity": [[] for _ in weights]})
        bucket["fitness"].append(fitness)
        bucket["runtime"].append(runtime)
        for i, value in enumerate(diversities):
            bucket["diversity"][i].append(value)

    ordered = [m for m in ("greedy", "pso", "hybrid", "random", "genetic") if m in collected]
    rows: list[DiversityPoint] = []
    for method in ordered:
        vals = collected[method]
        for i, weight in enumerate(weights):
            rows.append(DiversityPoint(
                dataset, graph_limit, params.max_nodes, method, len(seeds), weight,
                float(np.mean(vals["fitness"])),
                float(np.mean(vals["diversity"][i])), float(np.std(vals["diversity"][i])),
                float(np.mean(vals["runtime"]))))
    return rows


def diversity_to_csv(rows: list[DiversityPoint]) -> str:
    header = ("dataset,graph_limit,max_size,method,num_seeds,diversity_weight,"
              "fitness_mean,diversity_mean,diversity_std,runtime_mean")
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.dataset},{r.graph_limit},{r.max_size},{r.method},{r.num_seeds},"
            f"{r.diversity_weight},{r.fitness_mean:.6f},{r.diversity_mean:.6f},"
            f"{r.diversity_std:.6f},{r.runtime_mean:.6f}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ParetoPoint:
    """Multi-objective quality of one method's result pool."""
    dataset: str
    graph_limit: int
    max_size: int
    method: str
    num_seeds: int
    front_size: float          # mean number of non-dominated patterns found
    hvr_mean: float            # mean hypervolume ratio against the reference front
    hvr_std: float
    coverage_of_reference: float   # C(method, reference): share of the reference it covers
    best_fitness_mean: float   # the scalar view, for contrast
    runtime_mean: float


def _front_of(result, params: Params) -> list[tuple[float, ...]]:
    """The Pareto front of everything a run admitted (its pool, not just its Top-K)."""
    points = [objectives(c.evaluation, c.particle.pattern) for c in result.pool]
    return pareto_front(points) if points else []


def run_pareto_comparison(graphs: list[Graph], dataset: str, graph_limit: int,
                          seeds: list[int], params: Params,
                          include_exhaustive: bool = True,
                          methods: tuple[str, ...] = ("pso", "clpso", "hybrid",
                                                      "random", "genetic"),
                          workers: int = 1) -> list[ParetoPoint]:
    """Compare methods on multi-objective set quality rather than a single best score.

    Each run's whole candidate pool is reduced to its Pareto front over
    (WSUP, ATW, -size). The reference front is the exhaustive front when it is available --
    a true optimum, not merely the best known -- otherwise the union of every method's
    front. Reported per method: mean HVR, mean front size, and C(method, reference).
    """
    subset = graphs[:graph_limit]
    stats = build_statistics(subset)
    budget = params.swarm_size * (1 + params.iterations)

    jobs: list[tuple[str, int]] = [("greedy", -1)]
    if include_exhaustive:
        jobs.append(("exhaustive", -1))
    for method in methods:
        jobs.extend((method, seed) for seed in seeds)

    outcomes = run_front_jobs(jobs, subset, stats, params, budget, workers)

    fronts: dict[str, list[list[tuple[float, ...]]]] = {}
    scalars: dict[str, dict[str, list[float]]] = {}
    for method, _seed, front, fitness_value, runtime in outcomes:
        fronts.setdefault(method, []).append(front)
        bucket = scalars.setdefault(method, {"fitness": [], "runtime": []})
        bucket["fitness"].append(fitness_value)
        bucket["runtime"].append(runtime)

    if "exhaustive" in fronts:
        reference = fronts["exhaustive"][0]
    else:   # best-known front: the non-dominated union of everything any method found
        reference = pareto_front([p for runs in fronts.values()
                                 for front in runs for p in front])

    ordered = [m for m in ("exhaustive", "greedy", *methods) if m in fronts]
    rows: list[ParetoPoint] = []
    for method in ordered:
        runs = fronts[method]
        ratios = [hypervolume_ratio(front, reference) for front in runs]
        covers = [coverage(front, reference) for front in runs]
        rows.append(ParetoPoint(
            dataset, graph_limit, params.max_nodes, method, len(runs),
            float(np.mean([len(f) for f in runs])),
            float(np.mean(ratios)), float(np.std(ratios)),
            float(np.mean(covers)),
            float(np.mean(scalars[method]["fitness"])),
            float(np.mean(scalars[method]["runtime"]))))
    return rows


def pareto_to_csv(rows: list[ParetoPoint]) -> str:
    header = ("dataset,graph_limit,max_size,method,num_seeds,front_size,"
              "hvr_mean,hvr_std,coverage_of_reference,best_fitness_mean,runtime_mean")
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.dataset},{r.graph_limit},{r.max_size},{r.method},{r.num_seeds},"
            f"{r.front_size:.3f},{r.hvr_mean:.6f},{r.hvr_std:.6f},"
            f"{r.coverage_of_reference:.6f},{r.best_fitness_mean:.6f},{r.runtime_mean:.6f}")
    return "\n".join(lines) + "\n"
