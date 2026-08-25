"""Decide, cheaply, whether local improvement suffices on this dataset.

The enumerated measure in run_basins.py answers the question but cannot be used for
anything: it needs the full frequent pattern space, which is what mining exists to
avoid, and it is exactly what pdb_contact cannot supply. An elegant quantity no
practitioner can compute is not a selector.

The first attempt here estimated the enumerated flag by ascending from sampled starts
and asking whether the most-reached terminus was also the best-valued one. It is kept
in the history because its failure is the reason for the present design. Calibration
caught it on the first cell: BZR under sum has the flag off, its optimum draining 0.56
percent of the space, and the estimator said on. Nothing was miscoded. Forty ascents
reached six of the seventy-two attractors and never the optimum, so `best_seen` settled
on a lesser one, which was indeed the most-reached -- and the answer inverted.

The lesson generalises and no sample size repairs it: **an ascent-only probe cannot
detect that the optimum sits in a small basin, because it never sees the optimum.**

So this measures the thing the flag was standing in for, directly. Split one budget in
half. Spend one half ascending from dispersed starts, which is what an exploiting method
does; spend the other half sampling patterns globally, which is what ACO's construction
does. Then compare the best each half found:

    local_best >= global_best   local improvement suffices. Exploit.
    global_best >  local_best   the good patterns are somewhere ascent does not reach.
                                Sample globally.

That is self-consistent in a way the proxy was not. It never needs the true optimum,
only the two probes' own results, and it asks exactly the question the selector answers.
It is also honest about what it can be wrong about: with too small a budget both halves
find little and the verdict is noise, so the margin between them is reported and a
run that cannot separate them says so instead of guessing.

Calibration is refutable and prints its denominator. The enumerated flag is known on the
collections that can be enumerated; disagreement is reported per cell and exits non-zero.

    python scripts/estimate_basins.py --dataset pdb_contact --limit 200 --sizes 5 \
        --supports 2 --modes sum --budget 400 --suffix _pdb
    python scripts/estimate_basins.py --calibrate
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                             # noqa: E402

import run_weight_mode as rwm                                  # noqa: E402
from wfsmbench.core.signature import canonical_signature       # noqa: E402
from wfsmbench.io import load_dataset                          # noqa: E402
from wfsmbench.preprocess import build_statistics              # noqa: E402
from wfsmbench.pso.optimizer import (                          # noqa: E402
    Evaluator,
    _neighbors,
    _seed_particle,
)

RESULTS = ROOT / "results"
TOLERANCE = 1e-12

HEADER = ("dataset,graph_limit,max_size,min_support,weight_mode,budget,"
          "local_evaluations,global_evaluations,local_best,global_best,margin,"
          "flag_hat,ascents,attractors_seen,mean_path")

# The collections whose space can be enumerated, so the estimate has something to be
# checked against. Kept as (dataset, limit, suffix of the enumerated artefact).
ENUMERABLE = [
    ("Cuneiform", 60, "_cuneiform"),
    ("QM9_bond", 200, "_qm9"),
    ("BZR_bond", 200, "_BZRbond"),
    ("COX2_bond", 200, "_COX2bond"),
    ("DHFR_bond", 200, "_DHFRbond"),
    ("alchemy_full_bond", 200, "_alch"),
]


def disperse(particle, frequent, params, rng, steps):
    """Random single edits away from a seed, so starts are spread over the space.

    Ascending from single-edge patterns only would sample one corner of the space and
    report the basin of whatever sits above that corner. The walk is unbiased by
    fitness -- nothing is evaluated here -- so it costs no budget.
    """
    for _ in range(steps):
        options = _neighbors(particle, frequent, params)
        if not options:
            break
        particle = options[int(rng.integers(len(options)))]
    return particle


def ascend(particle, frequent, params, evaluator):
    """Steepest ascent until no neighbour is strictly fitter, or the budget runs out.

    Returns the terminal signature, its fitness and the number of steps. Neighbours are
    ordered by signature before comparison so ties resolve the same way on every run.
    """
    current = evaluator.score(particle)
    steps = 0
    while not evaluator.exhausted:
        options = sorted(_neighbors(current.particle, frequent, params),
                         key=lambda p: canonical_signature(p.pattern))
        best = None
        for option in options:
            if evaluator.exhausted:
                break
            scored = evaluator.score(option)
            if scored.fitness > current.fitness + TOLERANCE and (
                    best is None or scored.fitness > best.fitness + TOLERANCE):
                best = scored
        if best is None:
            break
        current, steps = best, steps + 1
    return canonical_signature(current.particle.pattern), current.fitness, steps


def local_probe(subset, stats, params, frequent, templates, rng, budget):
    """Ascend from dispersed starts until the budget is gone. What exploitation reaches."""
    evaluator = Evaluator(subset, stats, params, budget)
    best = float("-inf")
    ends, ascents, steps_total = set(), 0, 0
    while not evaluator.exhausted:
        start = _seed_particle(templates[int(rng.integers(len(templates)))], stats,
                               params, rng)
        start = disperse(start, frequent, params, rng,
                         int(rng.integers(0, params.max_edges)))
        signature, value, steps = ascend(start, frequent, params, evaluator)
        ends.add(signature)
        ascents += 1
        steps_total += steps
        best = max(best, value)
    return best, evaluator.spent, ascents, len(ends), (steps_total / ascents if ascents else 0.0)


def global_probe(subset, stats, params, frequent, templates, rng, budget):
    """Score patterns drawn from across the space, never following an improvement.

    This is the half that stands for a globally sampling method. Walk lengths are drawn
    over the whole size range rather than fixed, so the sample is not concentrated at one
    pattern size -- the objective is degenerate in size, and a fixed length would measure
    that degeneracy instead of the space.
    """
    evaluator = Evaluator(subset, stats, params, budget)
    best = float("-inf")
    while not evaluator.exhausted:
        particle = _seed_particle(templates[int(rng.integers(len(templates)))], stats,
                                  params, rng)
        particle = disperse(particle, frequent, params, rng,
                            int(rng.integers(0, params.max_edges)))
        best = max(best, evaluator.score(particle).fitness)
    return best, evaluator.spent


def estimate(subset, stats, params, budget: int, seed: int):
    frequent = set(stats.frequent_edge_templates(params.min_support))
    if not frequent:
        return None
    templates = sorted(frequent, key=repr)
    half = max(1, budget // 2)
    local_best, local_spent, ascents, seen, mean_path = local_probe(
        subset, stats, params, frequent, templates, np.random.default_rng(seed), half)
    global_best, global_spent = global_probe(
        subset, stats, params, frequent, templates, np.random.default_rng(seed + 1), half)
    if ascents == 0:
        return None
    return {
        "budget": budget,
        "local_evaluations": local_spent,
        "global_evaluations": global_spent,
        "local_best": local_best,
        "global_best": global_best,
        "margin": local_best - global_best,
        # Ties go to exploitation: equal results mean ascent lost nothing, which is the
        # cheaper choice and the one the mechanism predicts on a dominant optimum.
        "flag_hat": int(local_best >= global_best - TOLERANCE),
        "ascents": ascents,
        "attractors_seen": seen,
        "mean_path": mean_path,
    }


def run_one(dataset, limit, size, support, mode, budget, seed, rho):
    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs if limit is None else graphs[:limit]
    stats = build_statistics(subset)
    params = rwm.params_for(size, rwm.BUDGETS[0], mode, support, False, rho, 0.0)
    return estimate(subset, stats, params, budget, seed), len(subset)


def truth(suffix, dataset, mode):
    """The enumerated flag for one cell, or None when no artefact covers it."""
    path = RESULTS / f"basins{suffix}.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["policy"] == "strict" and row["dataset"] == dataset
                    and row["weight_mode"] == mode):
                return (int(row["optimum_is_largest"]),
                        float(row["basin_optimum_pct"]))
    return None


def calibrate(budget, seed):
    print(f"calibrating against the enumerated flag, budget {budget} per cell\n")
    print(f"  {'dataset':<20}{'mode':<12}{'flag':>6}{'flag_hat':>10}{'basin':>9}"
          f"{'local':>10}{'global':>10}{'margin':>10}   ")
    agree = compared = 0
    for dataset, limit, suffix in ENUMERABLE:
        for mode in ("sum", "discounted"):
            known = truth(suffix, dataset, mode)
            if known is None:
                print(f"  {dataset:<20}{mode:<12}   no enumerated artefact, skipped")
                continue
            got, _ = run_one(dataset, limit, 5, 2, mode, budget, seed, rwm.RHO)
            if got is None:
                print(f"  {dataset:<20}{mode:<12}   nothing frequent, skipped")
                continue
            compared += 1
            ok = got["flag_hat"] == known[0]
            agree += ok
            print(f"  {dataset:<20}{mode:<12}{known[0]:>6}{got['flag_hat']:>10}"
                  f"{known[1]:>8.2f}%{got['local_best']:>10.4f}{got['global_best']:>10.4f}"
                  f"{got['margin']:>+10.4f}   {'ok' if ok else 'WRONG'}")
    if compared == 0:
        print("\n[fail] nothing was compared; no enumerated artefact was found. "
              "Run stage 11 first -- a calibration over zero cells is not a passed one.")
        return 1
    print(f"\n  flag agrees on {agree}/{compared} cells")
    if agree < compared:
        print("[fail] the estimator disagrees with the enumerated flag; it cannot stand "
              "in for it until that is understood.")
        return 1
    print("[ok] the sampled flag matches the enumerated one everywhere it could be checked.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="run every enumerable collection and compare against the "
                         "enumerated flag; exits non-zero on any disagreement")
    ap.add_argument("--dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sizes", default="5")
    ap.add_argument("--supports", default="2")
    ap.add_argument("--modes", default="sum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=400,
                    help="distinct patterns scored in total, split evenly between the "
                         "local and the global probe")
    ap.add_argument("--rho", type=float, default=rwm.RHO)
    ap.add_argument("--suffix", default=None,
                    help="required unless --calibrate, so two runs cannot share a file")
    args = ap.parse_args(argv)

    if args.calibrate:
        return calibrate(args.budget, args.seed)
    if not args.dataset or args.suffix is None:
        ap.error("--dataset and --suffix are required unless --calibrate is given")

    rows = [HEADER]
    for size in [int(s) for s in args.sizes.split(",")]:
        for support in [int(s) for s in args.supports.split(",")]:
            for mode in args.modes.split(","):
                got, count = run_one(args.dataset, args.limit, size, support, mode,
                                     args.budget, args.seed, args.rho)
                if got is None:
                    print(f"  size {size} supp {support} [{mode}]: nothing frequent",
                          flush=True)
                    continue
                rows.append(
                    f"{args.dataset},{count},{size},{support},{mode},{got['budget']},"
                    f"{got['local_evaluations']},{got['global_evaluations']},"
                    f"{got['local_best']:.6f},{got['global_best']:.6f},"
                    f"{got['margin']:.6f},{got['flag_hat']},{got['ascents']},"
                    f"{got['attractors_seen']},{got['mean_path']:.3f}")
                print(f"  size {size} supp {support} [{mode}]: local {got['local_best']:.4f} "
                      f"({got['ascents']} ascents) vs global {got['global_best']:.4f}, "
                      f"margin {got['margin']:+.4f} -> flag_hat {got['flag_hat']}",
                      flush=True)

    out = RESULTS / f"basins_estimated{args.suffix}.csv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[done] {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
