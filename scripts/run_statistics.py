"""Statistical tests over the per-seed search results.

The comparison's headline claim is now a claim about a *ranking* -- that the order of the
methods changes when the objective stops being degenerate -- and a ranking read off six mean
values is not evidence. These are the tests the metaheuristic-comparison literature settled
on for exactly this shape of data (Demsar, JMLR 2006): all methods are run on the same seeds,
so every comparison is paired.

    Wilcoxon signed-rank, per (mode, budget), for every pair of methods, paired by seed.
      Holm-corrected within each cell, because 15 pairwise tests at alpha = 0.05 would
      otherwise produce a "significant" result by chance more often than not.

    Friedman across the six methods within a mode, blocked by (budget, seed). This asks
      whether the methods differ at all before any pair is inspected.

    Nemenyi critical difference for the post-hoc, reported as mean ranks and one CD value:
      two methods differ significantly exactly when their mean ranks are further apart
      than the CD.

    Spearman between the two modes' mean-rank vectors, which is the ranking-change claim
      stated as a number rather than as a reading of a table.

Ties matter here and are reported rather than hidden. Once a budget is generous enough that
several methods reach the optimum on most seeds, their paired differences are exactly zero,
Wilcoxon discards those pairs, and the effective sample size is far below the number of
seeds. A p-value from three non-tied pairs is not evidence of anything, so n_effective sits
next to every p.

    python scripts/run_statistics.py --runs results/weight_mode_runs_s5_practical.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

import numpy as np                                             # noqa: E402
from scipy.stats import (                                       # noqa: E402
    friedmanchisquare,
    rankdata,
    spearmanr,
    studentized_range,
    wilcoxon,
)

ALPHA = 0.05


def load(path: Path):
    """(mode, budget, method, seed) -> fitness, plus the observed method and budget order."""
    fitness, methods, budgets, modes = {}, [], [], []
    datasets = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["weight_mode"], int(row["budget"]), row["method"], int(row["seed"]))
            fitness[key] = float(row["fitness"])
            datasets.add(row.get("dataset", "?"))
            for value, seen in ((row["method"], methods), (int(row["budget"]), budgets),
                                (row["weight_mode"], modes)):
                if value not in seen:
                    seen.append(value)
    return fitness, methods, sorted(budgets), modes, datasets


def holm(pairs):
    """Holm-Bonferroni: sort ascending, compare p_i against alpha/(m-i), stop at the first
    failure. Uniformly more powerful than Bonferroni and just as safe."""
    ordered = sorted(pairs, key=lambda item: item[1])
    m = len(ordered)
    adjusted, running = [], 0.0
    for index, (label, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (m - index)))
        adjusted.append((label, p, running))
    return adjusted


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path,
                    default=RESULTS / "weight_mode_runs_s5_practical.csv",
                    help="per-seed CSV written by run_weight_mode.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the pairwise table (default alongside --runs)")
    args = ap.parse_args(argv)
    if not args.runs.exists():
        raise SystemExit(f"no per-seed results at {args.runs}")

    fitness, methods, budgets, modes, datasets = load(args.runs)
    seeds = sorted({key[3] for key in fitness})
    # The dataset is printed because it is the one thing a mislabelled run gets
    # wrong silently: a file named _qm9 once held Cuneiform results and nothing
    # at the point of reading said so.
    print(f"{args.runs.name}: dataset {'/'.join(sorted(datasets))}, "
          f"{len(modes)} modes, {len(budgets)} budgets, "
          f"{len(methods)} methods, {len(seeds)} seeds", flush=True)

    rows = ["weight_mode,budget,method_a,method_b,n_effective,wins_a,wins_b,ties,"
            "median_difference,p_value,p_holm,significant"]
    mean_ranks = {}

    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)

        # --- Friedman first: is there anything to look at? ------------------------
        blocks = [[fitness[(mode, budget, method, seed)] for method in methods]
                  for budget in budgets for seed in seeds]
        columns = list(zip(*blocks))
        statistic, p_friedman = friedmanchisquare(*columns)
        print(f"Friedman over {len(blocks)} blocks (budget x seed): "
              f"chi2 = {statistic:.1f}, p = {p_friedman:.3g}"
              f"{'' if p_friedman < ALPHA else '  -- no overall difference'}", flush=True)

        # --- mean ranks and the Nemenyi critical difference ------------------------
        ranks = np.array([rankdata([-value for value in block]) for block in blocks])
        means = ranks.mean(axis=0)
        mean_ranks[mode] = means
        k, n = len(methods), len(blocks)
        cd = float(studentized_range.ppf(1 - ALPHA, k, 1e6)) / np.sqrt(2) * \
            np.sqrt(k * (k + 1) / (6 * n))
        print(f"mean ranks (1 = best), Nemenyi CD = {cd:.3f}:", flush=True)
        for method, rank in sorted(zip(methods, means), key=lambda item: item[1]):
            print(f"    {method:<12} {rank:.3f}", flush=True)

        # --- pairwise Wilcoxon within each budget ---------------------------------
        for budget in budgets:
            tests = []
            detail = {}
            for a, b in combinations(methods, 2):
                xs = np.array([fitness[(mode, budget, a, seed)] for seed in seeds])
                ys = np.array([fitness[(mode, budget, b, seed)] for seed in seeds])
                differences = xs - ys
                nonzero = differences[differences != 0]
                wins_a = int((differences > 0).sum())
                wins_b = int((differences < 0).sum())
                ties = int((differences == 0).sum())
                if len(nonzero) == 0:
                    p = 1.0                      # identical on every seed
                else:
                    p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                tests.append((f"{a} vs {b}", p))
                detail[f"{a} vs {b}"] = (len(nonzero), wins_a, wins_b, ties,
                                         float(np.median(differences)))
            significant = []
            for label, p, p_adj in holm(tests):
                n_eff, wins_a, wins_b, ties, median = detail[label]
                a, b = label.split(" vs ")
                is_sig = p_adj < ALPHA and n_eff > 0
                rows.append(f"{mode},{budget},{a},{b},{n_eff},{wins_a},{wins_b},{ties},"
                            f"{median:.6f},{p:.6g},{p_adj:.6g},{int(is_sig)}")
                if is_sig:
                    better, worse = (a, b) if median > 0 else (b, a)
                    significant.append(f"{better}>{worse} (n={n_eff}, p={p_adj:.2g})")
            print(f"  budget {budget:>4}: "
                  f"{len(significant)}/{len(tests)} pairs differ after Holm", flush=True)
            for line in significant:
                print(f"      {line}", flush=True)

    # --- the ranking-change claim, as a number --------------------------------
    if len(modes) == 2:
        first, second = modes
        rho, p_rho = spearmanr(mean_ranks[first], mean_ranks[second])
        print(f"\n=== ranking change: {first} vs {second} ===", flush=True)
        print(f"{'method':<12}{first:>12}{second:>14}   shift", flush=True)
        order_a = rankdata(mean_ranks[first])
        order_b = rankdata(mean_ranks[second])
        for index, method in enumerate(methods):
            shift = int(order_a[index] - order_b[index])
            print(f"{method:<12}{mean_ranks[first][index]:>12.3f}"
                  f"{mean_ranks[second][index]:>14.3f}   "
                  f"{'+' if shift > 0 else ''}{shift}", flush=True)
        print(f"\nSpearman between the two orderings: rho = {rho:.3f}, p = {p_rho:.3g}",
              flush=True)
        print("rho near 1 would mean the objective's defect did not change the ranking; "
              "the further below 1, the more of the reported ordering belonged to the "
              "objective rather than to the methods.", flush=True)

    out = args.out or RESULTS / args.runs.name.replace("weight_mode_runs",
                                                       "statistics_pairwise")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n[done] {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake      # Windows: no sleep while the stage runs
    keep_awake()
    raise SystemExit(main(sys.argv[1:]))
