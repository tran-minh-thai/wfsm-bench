"""Is the objective degenerate, and if so what makes it degenerate?

Two things are swept: the weight mode, and the support threshold.

**The weight mode.** Every result this project produced was measured under
`weight_mode="sum"`; `per_edge` had been implemented and unit-tested since 484c221 and no
experiment ever used it. The first run of this script settled that question and the answer
was that neither mode gives a problem worth searching. On Cuneiform (n=60, size 4, 712
patterns) the best F reachable at each edge count was:

    sum       1:0.0420  2:0.0737  3:0.1047  4:0.1213   strictly rising -> optimum at the bound
    per_edge  1:0.4738  2:0.3828  3:0.3242  4:0.2723   strictly falling -> optimum a single edge

Strictly monotone at every step in both, with no turning point. `per_edge` does not create
an interior optimum; it moves the degeneracy to the other end, and its optimum is a
one-edge pattern, which is a trivial mining result.

**The support threshold**, which is the more likely culprit and the reason this sweep
exists. WSUP = support x ATW. `min_support` was 2 out of 60 graphs -- 3.3%, where real
frequent-subgraph mining uses 10-50% -- so the threshold barely binds, support hardly falls
as a pattern grows, and WSUP therefore tracks ATW, which under "sum" rises with size. For
WSUP to peak somewhere in the middle, support has to genuinely collapse as patterns grow,
and that only happens once the threshold bites. Thresholds are swept as *fractions* of the
database so the two datasets are comparable at different `graph_limit`s.

Three outcomes are possible and all three are worth reporting: a peak appears in
`best_f_by_edges`, and there is a real optimization problem after all; the pool shrinks but
the frontier stays monotone, and the objective needs redesigning rather than retuning; or
the pool empties at realistic thresholds, and these datasets are too sparse for mining at
the thresholds the field actually uses.

`results/weight_mode_landscape.csv` -- the diagnostic, from exhaustive enumeration:

    best_f_by_edges     the best F reachable at each edge count. Read this column first: it
                        shows the shape of the objective directly. Rising to the bound means
                        "bigger is better", falling from 1 means "smaller is better", and a
                        peak in between is the interior optimum a population search needs.
    optimum_edges       where the peak actually is, against the bound.
    spearman_f_edges    rank correlation of F with edge count over the whole pool. Near +1
                        or -1 is a monotone objective; near 0 is a real trade-off. Read it
                        alongside best_f_by_edges, not instead of it: the cloud of patterns
                        can be noisy while the frontier that decides the optimum is not.
    local_optima        pool patterns with no fitter single-edit neighbour. One means the
                        landscape is unimodal and a hill-climber cannot be beaten.
    fitness_levels      distinct F values as a fraction of the pool. Well below 1 means large
                        plateaus, where gbest is decided by tie-breaking rather than search.
    frequent_templates  } these two exist to check the reading of local_optima. Under
    pool_at_bound       } "sum" a pattern is a local optimum when it cannot grow, so the
                        count should track pool_at_bound; under "per_edge" it is one that
                        cannot shrink, so it should track frequent_templates. If they match,
                        the local optima are an artefact of the bounds and of monotonicity
                        rather than evidence of a rugged landscape.
    max_edges_in_pool   the largest pattern that stays frequent. Below the size bound, the
                        support threshold is binding -- which is the condition this sweep is
                        looking for.

`results/weight_mode_methods.csv` -- whether the ranking of the searches changes, at matched
budgets against the exhaustive optimum for each (mode, threshold).

    python scripts/run_weight_mode.py --preset quick --landscape-only   # start here
    python scripts/run_weight_mode.py --preset balanced --workers 10

The search arm runs at every swept threshold, so its cost is multiplied by the number of
thresholds; --landscape-only is the cheap first pass and on its own answers whether any
threshold makes the problem non-degenerate. Higher thresholds are cheaper, not dearer: they
shrink the pool.

Run it as a script, never by pasting into a prompt: the workers are started with spawn on
Windows and re-import this file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                          # noqa: E402

from wfsmbench.baselines import (                            # noqa: E402
    mine_aco,
    mine_exhaustive,
    mine_genetic,
    mine_local_search,
    mine_random,
)
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params    # noqa: E402
from wfsmbench.core.signature import canonical_signature     # noqa: E402
from wfsmbench.io import load_dataset                        # noqa: E402
from wfsmbench.measure import fitness as score_fitness       # noqa: E402
from wfsmbench.preprocess import build_statistics            # noqa: E402
from wfsmbench.pso import optimize, optimize_clpso, optimize_multiswarm   # noqa: E402
# The single-edit neighbourhood the searches actually move through -- grow by any frequent
# template, or shrink an edge. Counting local optima against anything else would measure a
# landscape none of the methods traverse. clpso and multiswarm import it the same way.
from wfsmbench.pso.optimizer import _neighbors               # noqa: E402

RESULTS = ROOT / "results"
SWARM = 20
MODES = ["sum", "per_edge", "discounted"]
# The band where the methods actually separate. Measured at size 6 / 50% threshold, the
# old grid spent two of its three points outside it: at 540 (12.5% of the space) three
# methods were already tied at the optimum, and 1800 covered the space several times over.
# Everything below stays at three swarm generations or more, so a budget of 30 -- barely
# one generation for a swarm of 20 -- is deliberately excluded as a degenerate setting
# rather than a tight one.
BUDGETS = [60, 120, 180, 240]
# local_search is the single-solution baseline: nothing else here keeps just one solution
# and walks uphill, and on a landscape with 22 local optima that is the method most likely
# to make the population machinery look unnecessary. It belongs in the table for that
# reason, not despite it.
METHODS = ["pso", "clpso", "multiswarm", "aco", "genetic", "local_search", "random"]
# The discount factor for weight_mode="discounted"; calibrated per dataset, see config.py.
RHO = Params.rho


def calibrated_rho(dataset: str) -> float | None:
    """The largest rho measured stable for this collection, read from the artefacts.

    The discount is a per-collection quantity: the criterion is that the optimum stays
    interior and on the same pattern as the size bound widens, and where that holds
    depends on how fast support falls with pattern size, which is a property of the data.
    Cuneiform calibrates to 0.5 and QM9 to 0.6, so a shared default is right for one of
    them by accident.

    Resolved here rather than passed in because a caller that forgets the flag gets a
    number that looks fine. That happened: two stages ran QM9 under discounting at the
    default while another used the calibrated value, and the two searched different
    objectives for four days before a determinism check found 280 disagreeing rows.
    Making the calibrated value the default, and refusing to run without one, removes
    the chance to forget.

    The largest stable rho is taken because stability is a floor -- everything below the
    turnover stays stable and discounts harder than necessary, so the largest is the
    least intervention that satisfies the criterion.
    """
    best = None
    for path in sorted(RESULTS.glob("degeneracy_rho*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") == dataset and row.get("stable") == "1":
                    value = float(row["rho"])
                    best = value if best is None else max(best, value)
    return best


def resolve_rho(dataset: str, modes: list[str], explicit: float | None) -> float:
    """The discount to search under, or a refusal saying what to run first."""
    if explicit is not None:
        return explicit
    if "discounted" not in modes:
        return RHO                      # unused by these modes; any value is inert
    value = calibrated_rho(dataset)
    if value is None:
        raise SystemExit(
            f"[refused] no calibrated rho for {dataset}. Running discounted at the "
            f"default would search an objective never justified for this collection. "
            f"Calibrate first:\n"
            f"    python scripts/run_degeneracy.py --dataset {dataset} --sizes 4,5 "
            f"--supports 2 --calibrate-rho --suffix _<name>")
    return value

# Support thresholds as a fraction of the database, so the datasets are comparable at
# different graph_limits. 0.03 reproduces the min_support=2 every previous result used;
# the rest span the range frequent-subgraph mining normally works in. 0.60 and 0.70 are past
# that range and are here because the first sweep found the interior optimum only at 0.50,
# the most extreme threshold it tested -- a peak that appears at the edge of the sweep has
# not been bounded, and the question is whether it sharpens or disappears beyond it.
SUPPORT_FRACTIONS = [0.03, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70]

PRESETS = {
    # dataset -> (graph_limit, [max sizes]); seeds applies to the stochastic methods
    "quick":    {"seeds": 5, "plan": {"Cuneiform": (60, [4])}},
    "balanced": {"seeds": 5, "plan": {"Cuneiform": (60, [4, 5]), "BZR_MD": (20, [3, 4])}},
    # Does the interior optimum survive as the space grows? At size 4 the pool was 365
    # patterns at the 50% threshold, which is smaller than two of the three budgets -- an
    # interior optimum is necessary for the searches to have anything to do, but not
    # sufficient while the budget still covers the whole space. Cuneiform only: BZR_MD is
    # dense enough that almost everything is frequent, so a high threshold prunes little
    # there and the enumeration stays expensive at every level.
    "deep":     {"seeds": 5, "plan": {"Cuneiform": (60, [4, 5, 6])}},
}

LANDSCAPE_HEADER = (
    "dataset,graph_limit,max_size,support_fraction,min_support,weight_mode,pool_size,"
    "frequent_templates,pool_at_bound,max_edges_in_pool,optimum_fitness,optimum_multiplicity,optimum_nodes,"
    "optimum_edges,optimum_support,size_bound,optimum_is_interior,spearman_f_edges,"
    "local_optima,fitness_levels,local_optima_true,fitness_levels_true,fitness_quantum,best_f_by_edges")
RUN_HEADER = ("dataset,graph_limit,max_size,min_support,weight_mode,budget,method,seed,fitness,found_optimum,prune_infrequent,rho")
METHOD_HEADER = (
    "dataset,graph_limit,max_size,support_fraction,min_support,weight_mode,budget,method,"
    "seeds,fitness_mean,fitness_std,gap_pct,hits,hits_fitness_tie,optimum_multiplicity,"
    "prune_infrequent,rho")

_STATE: dict = {}


def supports_for(graph_limit: int) -> list[tuple[float, int]]:
    """(effective fraction, absolute threshold) pairs for one database size.

    Floored at 2, since a pattern in a single graph is not frequent under any reading, and
    deduplicated: on a small database several requested fractions round to one threshold.

    The fraction reported back is the one the threshold actually represents,
    ``absolute / graph_limit``, not the one that was asked for. On 20 graphs both 3% and 10%
    round to a threshold of 2, and recording that row as 3% would understate the constraint
    the run was really under by a factor of three.
    """
    absolutes = sorted({max(2, round(fraction * graph_limit))
                        for fraction in SUPPORT_FRACTIONS})
    return [(absolute / graph_limit, absolute) for absolute in absolutes]


def group_by_aggregation(modes: list[str], rho: float) -> dict[str, list[str]]:
    """Modes grouped by the W(e) they need, so one enumeration serves a whole group.

    Enumeration and matching are the expensive half, and what they produce -- the pool and
    each pattern's Evaluation -- depends only on the aggregation, not on how the fitness
    later uses it. "sum" and "per_edge" both aggregate by plain sum; "per_edge" divides by
    the edge count afterwards, in the fitness. "discounted" aggregates differently and so
    needs its own pool.
    """
    groups: dict[str, list[str]] = {}
    for mode in modes:
        key = f"discounted:{rho}" if mode == "discounted" else "plain"
        groups.setdefault(key, []).append(mode)
    return groups


def params_for(size: int, budget: int, mode: str, min_support: int,
               prune: bool = False, rho: float = RHO, quantum: float = 0.0) -> Params:
    """The configuration under test. Only ``weight_mode`` and ``min_support`` vary -- the
    size bounds, the budget-dependent settings and the fitness anchor are held fixed."""
    rounds = max(1, budget // SWARM)
    return Params(
        max_nodes=size, max_edges=size, min_support=min_support,
        size_reference=STUDY_SIZE_REFERENCE,
        weight_mode=mode, rho=rho, prune_infrequent=prune, fitness_quantum=quantum,
        swarm_size=SWARM, iterations=rounds - 1,
        clpso_refresh_gap=max(1, rounds // 8),
    )


# --- landscape diagnostics ----------------------------------------------------

def _ranks(values: list[float]) -> np.ndarray:
    """Ranks with ties averaged, which is what Spearman needs. Done by hand because scipy
    is not a dependency and adding one for a single correlation is not worth it."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)
    # Average the ranks within each group of equal values.
    start = 0
    ordered = array[order]
    for index in range(1, len(array) + 1):
        if index == len(array) or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0
            start = index
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. 0.0 when either side is constant, where it is undefined."""
    if len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def enumerate_pool(subset, stats, params: Params):
    """The frequent pattern space at this size bound and support threshold.

    The space does not depend on the weight mode: membership is decided by support and the
    size bounds, and the per-pattern Evaluation (support, ATW, WSUP) is what the matcher
    produces, before any mode is applied. Only the scalarization into F differs. So the
    expensive half -- enumeration and subgraph matching -- runs once per threshold and each
    mode re-scores the same candidates, which halves that cost and makes the two modes exact:
    identical patterns, identical evaluations, nothing differing but the objective. The
    threshold does change the space, so it cannot share an enumeration the same way.
    """
    return mine_exhaustive(subset, stats, params).pool


def landscape(pool, stats, params: Params, tolerance: float = 1e-12) -> dict:
    """Describe the shape of ``params``'s objective over an already-enumerated pool."""
    if not pool:
        return {}

    # Two scorings of the same evaluations, so no pattern is matched twice. `scored` is
    # what a search navigates; `true_scored` is what its answer is judged on. With no
    # quantum the two are identical and everything below reduces to the previous
    # behaviour -- which is why the reference optimum is taken from the true values in
    # both cases rather than only when a quantum is set.
    true_params = replace(params, fitness_quantum=0.0)
    scored, true_scored = {}, {}
    for candidate in pool:
        pattern = candidate.particle.pattern
        scored[candidate.signature] = score_fitness(
            candidate.evaluation, pattern.num_nodes(), pattern.num_edges(), stats, params)
        true_scored[candidate.signature] = score_fitness(
            candidate.evaluation, pattern.num_nodes(), pattern.num_edges(), stats, true_params)

    fitnesses = [scored[c.signature] for c in pool]
    true_fitnesses = [true_scored[c.signature] for c in pool]
    edge_counts = [c.particle.pattern.num_edges() for c in pool]

    best_by_edges: dict[int, float] = {}
    for fitness, edges in zip(fitnesses, edge_counts):
        if edges not in best_by_edges or fitness > best_by_edges[edges]:
            best_by_edges[edges] = fitness

    # A pattern is a local optimum when no single edit reaches a fitter one. Neighbours
    # outside the pool are infrequent, so they are not reachable results and do not count.
    frequent = set(stats.frequent_edge_templates(params.min_support))

    def count_optima(table: dict[str, float]) -> int:
        # Counted separately for the searched objective and the true one. Under a quantum
        # the two differ, and by how much is the point: rounding merges neighbours into
        # plateaus, so a pattern with a slightly fitter neighbour stops being distinguishable
        # from it. That change is a consequence of the manipulation, not a confound -- but it
        # means the experiment varies plateau structure as a whole, not the level count in
        # isolation, and the two columns are what let a reader see that.
        found = 0
        for candidate in pool:
            value = table[candidate.signature]
            best_neighbour = max(
                (table[sig] for sig in
                 (canonical_signature(n.pattern)
                  for n in _neighbors(candidate.particle, frequent, params))
                 if sig in table),
                default=None)
            if best_neighbour is None or best_neighbour <= value + tolerance:
                found += 1
        return found

    optima = count_optima(scored)
    true_optima = optima if params.fitness_quantum <= 0.0 else count_optima(true_scored)

    best = max(pool, key=lambda c: true_scored[c.signature])
    # Every pattern that attains the optimal fitness, not just one of them. A run counts as
    # having found the optimum when it returns one of these, which is a different and much
    # stricter claim than returning something that scores the same -- and on a landscape this
    # flat the two diverge. How many there are is reported so the claim can be calibrated:
    # a "hit" on an optimum shared by one pattern and a hit on one shared by fifty are not
    # the same achievement.
    optimum_signatures = {c.signature for c in pool
                          if true_scored[c.signature] >= true_scored[best.signature] - tolerance}
    best_pattern = best.particle.pattern
    return {
        "pool_size": len(pool),
        "frequent_templates": len(frequent),
        "pool_at_bound": sum(1 for e in edge_counts if e >= params.max_edges),
        "max_edges_in_pool": max(edge_counts),
        "optimum_fitness": true_scored[best.signature],
        "optimum_signatures": optimum_signatures,
        "optimum_multiplicity": len(optimum_signatures),
        "optimum_nodes": best_pattern.num_nodes(),
        "optimum_edges": best_pattern.num_edges(),
        "optimum_support": best.evaluation.support,
        "optimum_is_interior": int(1 < best_pattern.num_edges() < params.max_edges),
        "spearman_f_edges": spearman(fitnesses, [float(e) for e in edge_counts]),
        "local_optima": optima,
        "fitness_levels": len({round(f, 12) for f in fitnesses}) / len(pool),
        # The same two quantities on the objective as it really is. Equal to the pair
        # above whenever no quantum is set; when one is, the gap between the columns is
        # the size of the manipulation, measured rather than assumed from the grid width.
        "local_optima_true": true_optima,
        "fitness_levels_true": len({round(f, 12) for f in true_fitnesses}) / len(pool),
        "best_f_by_edges": "|".join(
            f"{edges}:{best_by_edges[edges]:.4f}" for edges in sorted(best_by_edges)),
    }


# --- the searches -------------------------------------------------------------

def _init(subset, stats) -> None:
    _STATE.update(subset=subset, stats=stats)


def _job(spec):
    method, size, budget, mode, support, seed, prune, rho, quantum = spec
    subset, stats = _STATE["subset"], _STATE["stats"]
    params = params_for(size, budget, mode, support, prune, rho, quantum)
    if method == "pso":
        result = optimize(subset, stats, params, seed, max_evaluations=budget)
    elif method == "clpso":
        result = optimize_clpso(subset, stats, params, seed, max_evaluations=budget)
    elif method == "multiswarm":
        result = optimize_multiswarm(subset, stats, params, seed, max_evaluations=budget)
    elif method == "aco":
        result = mine_aco(subset, stats, params, seed, max_evaluations=budget)
    elif method == "genetic":
        result = mine_genetic(subset, stats, params, seed, max_evaluations=budget)
    elif method == "local_search":
        result = mine_local_search(subset, stats, params, seed, max_evaluations=budget)
    elif method == "random":
        result = mine_random(subset, stats, params, seed, budget)
    else:
        raise ValueError(f"unknown method: {method}")
    # Reported on the true objective, never on the searched one. Under a quantum the
    # search cannot tell the members of a plateau apart and returns an arbitrary one; what
    # that pattern is actually worth is the measurement, and scoring it again with the
    # quantum off costs nothing because the match is already done. With no quantum this is
    # the same number as before.
    best = result.gbest
    if best is None:
        score = 0.0
    else:
        pattern = best.particle.pattern
        score = score_fitness(best.evaluation, pattern.num_nodes(), pattern.num_edges(),
                              stats, replace(params, fitness_quantum=0.0))
    return (method, size, budget, mode, support, seed, score,
            best.signature if best else None)


class Sink:
    """A CSV written row by row and flushed as it goes.

    The enumeration at the larger sizes runs for hours, and a run that buffered its rows and
    wrote them at the end lost every completed size when it was killed partway through. Rows
    are independent, so there is no reason to hold them: each one is useful on its own, and
    an interrupted run should leave behind everything it had finished.
    """

    def __init__(self, path: Path, header: str) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._handle.write(header + "\n")
        self._handle.flush()

    def write(self, line: str) -> None:
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes; 0 selects one per core, less one")
    ap.add_argument("--landscape-only", action="store_true",
                    help="skip the searches; the diagnostic alone answers whether any "
                         "threshold makes the objective non-degenerate, and it is much "
                         "the cheaper half")
    # The diagnostic sweeps broadly to find where the objective stops being degenerate;
    # once it has, the searches are worth running only at that cell. These narrow the run
    # to it rather than paying for the whole grid again.
    ap.add_argument("--sizes", help="comma-separated max sizes, overriding the preset")
    ap.add_argument("--supports", help="comma-separated absolute thresholds, overriding "
                                       "the fractional sweep")
    ap.add_argument("--modes", help=f"comma-separated weight modes (default {','.join(MODES)})")
    ap.add_argument("--budgets", help=f"comma-separated budgets (default "
                                      f"{','.join(str(b) for b in BUDGETS)})")
    ap.add_argument("--seeds", type=int, help="seed count, overriding the preset")
    ap.add_argument("--dataset",
                    help="dataset JSON stem in data/, replacing the preset's plan. "
                         "Without it a run silently uses whatever dataset the preset "
                         "names, which is how a QM9 stage once produced Cuneiform "
                         "numbers labelled as QM9.")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many graphs before taking --limit. QM9 and alchemy are "
                         "sorted by molecule size and hold six figures of graphs, so a later "
                         "slice is a different instance -- different molecules, pool, optimum "
                         "and basin -- drawn from the same population")
    ap.add_argument("--limit", type=int, default=None,
                    help="how many graphs; needed with --dataset unless the preset "
                         "already names that dataset")
    ap.add_argument("--quantum", type=float, default=0.0,
                    help="round the objective the search navigates onto this grid, "
                         "leaving the true objective for reporting; 0 disables it")
    ap.add_argument("--rho", type=float, default=None,
                    help="discount factor for weight_mode=discounted. Calibrated per\n"
                         "dataset -- see run_degeneracy.py --calibrate-rho -- so a run on\n"
                         "a dataset other than the one config.rho was fitted to must pass\n"
                         "its own value, or it measures the wrong objective.")
    ap.add_argument("--prune", action="store_true",
                    help="skip patterns that anti-monotonicity proves cannot be frequent; a\n"
                         "condition to compare against, not a default (see config.py)")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filenames, so a targeted run does not "
                         "overwrite the sweep it was derived from")
    args = ap.parse_args(argv)
    cfg = PRESETS[args.preset]
    plan = dict(cfg["plan"])
    if args.dataset:
        known = plan.get(args.dataset)
        limit = args.limit if args.limit is not None else (known[0] if known else None)
        if limit is None:
            raise SystemExit(
                f"--dataset {args.dataset} is not in preset {args.preset!r}, so "
                f"--limit is required")
        plan = {args.dataset: (limit, known[1] if known else [4])}
    elif args.limit is not None:
        plan = {name: (args.limit, sizes) for name, (_l, sizes) in plan.items()}
    seeds = list(range(args.seeds if args.seeds is not None else cfg["seeds"]))
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    modes = args.modes.split(",") if args.modes else list(MODES)
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else BUDGETS
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"unknown weight mode: {mode!r}; expected one of {MODES}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    landscape_csv = Sink(
        RESULTS / f"weight_mode_landscape{args.suffix}.csv", LANDSCAPE_HEADER)
    method_csv = (None if args.landscape_only
                  else Sink(RESULTS / f"weight_mode_methods{args.suffix}.csv",
                            METHOD_HEADER))
    # Per-seed rows as well as the aggregate. Every paired test -- Wilcoxon across seeds,
    # Friedman across methods -- needs the individual runs, and a mean with a standard
    # deviation cannot be un-aggregated back into them.
    run_csv = (None if args.landscape_only
               else Sink(RESULTS / f"weight_mode_runs{args.suffix}.csv", RUN_HEADER))
    try:

        for name, (limit, sizes) in plan.items():
            # Resolved per collection, inside the loop: the discount is a property of
            # the data, so one value cannot serve two datasets, and a plan may hold more
            # than one.
            rho = resolve_rho(name, modes, args.rho)
            if args.rho is None and "discounted" in modes:
                print(f"  {name}: rho = {rho} (calibrated)", flush=True)
            graphs, _ = load_dataset(ROOT / "data" / f"{name}.json")
            subset = graphs[args.offset:args.offset + limit] if limit else graphs[args.offset:]
            stats = build_statistics(subset)
            if args.sizes:
                sizes = [int(s) for s in args.sizes.split(",")]
            if args.supports:
                # Given as absolute thresholds; the fraction each represents follows from
                # the database size, the same way supports_for derives it.
                supports = [(int(s) / limit, int(s)) for s in args.supports.split(",")]
            else:
                supports = supports_for(limit)
            print(f"\n=== {name}  n={limit}  seeds={len(seeds)}  workers={workers} ===",
                  flush=True)
            print("  thresholds: "
                  + ", ".join(f"{f:.0%}->{a}" for f, a in supports), flush=True)

            # (optimal fitness, the signatures attaining it) per cell.
            optima: dict[tuple[int, int, str], tuple[float, set]] = {}
            for size in sizes:
                for fraction, support in supports:
                    # Enumerated once per *aggregation*, not once per mode. "sum" and
                    # "per_edge" share an Evaluation -- per_edge does its division later, in
                    # the fitness -- so one enumeration serves both, and on the size-6 cells
                    # that is two and a half hours not spent twice. "discounted" changes W(e)
                    # itself, so its ATW is a mean of discounted sums and its Evaluation is a
                    # different object; scoring a pool built under one aggregation against
                    # another's anchors would be wrong, so it gets its own.
                    for aggregation, group in group_by_aggregation(modes, rho).items():
                        params = params_for(size, budgets[0], group[0], support,
                                            args.prune, rho, args.quantum)
                        started = time.perf_counter()
                        pool = enumerate_pool(subset, stats, params)
                        elapsed = time.perf_counter() - started
                        shared = "/".join(group)
                        if not pool:
                            print(f"  size {size}  supp {support:>3} ({fraction:.0%}) "
                                  f"[{shared}]: pool empty -- nothing is frequent at this "
                                  f"threshold [{elapsed:.0f}s]", flush=True)
                            continue
                        print(f"  size {size}  supp {support:>3} ({fraction:.0%}) "
                              f"[{shared}]: enumerated {len(pool)} patterns "
                              f"[{elapsed:.0f}s]", flush=True)
                        for mode in group:
                            params = params_for(size, budgets[0], mode, support,
                                                args.prune, rho, args.quantum)
                            info = landscape(pool, stats, params)
                            optima[(size, support, mode)] = (info["optimum_fitness"],
                                                            info["optimum_signatures"])
                            landscape_csv.write(
                                f'{name},{limit},{size},{fraction},{support},{mode},'
                                f'{info["pool_size"]},{info["frequent_templates"]},'
                                f'{info["pool_at_bound"]},{info["max_edges_in_pool"]},'
                                f'{info["optimum_fitness"]:.6f},'
                                f'{info["optimum_multiplicity"]},{info["optimum_nodes"]},'
                                f'{info["optimum_edges"]},{info["optimum_support"]},{size},'
                                f'{info["optimum_is_interior"]},{info["spearman_f_edges"]:.4f},'
                                f'{info["local_optima"]},{info["fitness_levels"]:.4f},'
                                f'{info["local_optima_true"]},'
                                f'{info["fitness_levels_true"]:.4f},'
                                f'{params.fitness_quantum:g},'
                                f'{info["best_f_by_edges"]}')
                            shape = ("INTERIOR" if info["optimum_is_interior"]
                                     else "at the bound" if info["optimum_edges"] >= size
                                     else "at the floor")
                            binding = "binding" if info["max_edges_in_pool"] < size else "slack"
                            print(f'      [{mode:>8}] optimum {info["optimum_fitness"]:.4f} at '
                                  f'{info["optimum_edges"]}e ({shape}), threshold {binding}, '
                                  f'rho={info["spearman_f_edges"]:+.2f}, '
                                  f'{info["local_optima"]} local optima '
                                  f'(bound {info["pool_at_bound"]}, templates '
                                  f'{info["frequent_templates"]}), '
                                  f'{info["fitness_levels"]:.2f} levels', flush=True)
                            print(f'          best F by edge count: '
                                  f'{info["best_f_by_edges"]}', flush=True)

            if args.landscape_only:
                continue

            jobs = [(m, size, budget, mode, support, seed, args.prune, rho, args.quantum)
                    for size in sizes for _f, support in supports for mode in modes
                    for budget in budgets for m in METHODS for seed in seeds
                    if (size, support, mode) in optima]
            print(f"  {len(jobs)} search runs to do", flush=True)
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                     initargs=(subset, stats)) as pool_exec:
                outcomes = list(pool_exec.map(_job, jobs, chunksize=4))
            print(f"  finished in {time.perf_counter() - started:.0f}s", flush=True)

            collected: dict = {}
            for method, size, budget, mode, support, seed, fitness, sig in outcomes:
                collected.setdefault((size, support, mode, budget, method),
                                     []).append((fitness, sig))
                found = int(sig in optima[(size, support, mode)][1]) \
                    if (size, support, mode) in optima else 0
                run_csv.write(f"{name},{limit},{size},{support},{mode},{budget},{method},"
                              f"{seed},{fitness:.6f},{found},{int(args.prune)},{rho}")

            for size in sizes:
                for fraction, support in supports:
                    for mode in modes:
                        entry = optima.get((size, support, mode))
                        if entry is None:
                            continue
                        optimum, optimum_signatures = entry
                        print(f"\n  size {size} supp {support} [{mode}]"
                              + "".join(m.rjust(16) for m in METHODS), flush=True)
                        for budget in budgets:
                            cells = []
                            for method in METHODS:
                                runs = collected[(size, support, mode, budget, method)]
                                values = np.array([f for f, _s in runs])
                                # A hit is returning a pattern that *is* an optimum, not one
                                # that merely scores like one. On a landscape where a few
                                # dozen distinct fitness values are shared by thousands of
                                # patterns those are very different claims, and the old
                                # fitness-tie count is kept beside it so the gap between them
                                # is visible rather than argued about.
                                hits = sum(1 for _f, s in runs if s in optimum_signatures)
                                ties = int((values >= optimum - 1e-9).sum())
                                gap = (optimum - values.mean()) / abs(optimum) * 100
                                method_csv.write(
                                    f"{name},{limit},{size},{fraction},{support},{mode},"
                                    f"{budget},{method},{len(seeds)},{values.mean():.6f},"
                                    f"{values.std():.6f},{gap:.4f},{hits},{ties},"
                                    f"{len(optimum_signatures)},{int(args.prune)},{rho}")
                                # gap / patterns-that-are-an-optimum / fitness-ties
                                cells.append(
                                    f"{gap:5.2f}%/{hits:02d}/{ties:02d}".rjust(16))
                            print(f"  {budget:>7}" + "".join(cells), flush=True)

    finally:
        landscape_csv.close()
        if method_csv is not None:
            method_csv.close()
        if run_csv is not None:
            run_csv.close()
    print(f"\n[done] {landscape_csv.path.relative_to(ROOT)}", flush=True)
    if method_csv is not None:
        print(f"[done] {method_csv.path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake      # Windows: no sleep while the stage runs
    keep_awake()
    raise SystemExit(main(sys.argv[1:]))
