"""Friedman and Nemenyi with independent blocks: one block per seed instead of per (budget, seed).

The rank table blocks the Friedman test by (budget, seed): 4 budgets x 20 seeds = 80 blocks per
dataset and mode. Runs that share a seed at different budgets start from the same random stream,
so those 80 blocks are not independent, and the test's p-value and the Nemenyi critical
difference (CD) both assume they are. This script recomputes the analysis with the seed as the
block: within each seed, the methods are ranked at every budget and the ranks are averaged over
the budgets; the Friedman test then runs over the 20 seed blocks, and the CD uses n = 20.

It first recomputes the CURRENT analysis (80 blocks) and refuses unless the mean ranks and the
largest p-value match the tracked statistics logs, so a difference between the two analyses is
the change of block, not a different reading of the data.

For each analysis it reports, per dataset and mode: the Friedman p-value, the CD, whether the
best method is ahead of the second by more than the CD; and over each mode, every pair of methods
whose order reverses between two datasets with a gap above the CD in both.

Counts and p-values only, no timings.

    python scripts/friedman_seed_blocks.py --out results-invariant/friedman_seed_blocks.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHA = 0.05

sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import STATS_LOG  # noqa: E402  the collection -> run-file map of the rank table


def load(path: Path):
    fitness, methods, budgets, modes = {}, [], set(), []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["weight_mode"], int(row["budget"]), row["method"], int(row["seed"]))
            fitness[key] = float(row["fitness"])
            if row["method"] not in methods:
                methods.append(row["method"])
            if row["weight_mode"] not in modes:
                modes.append(row["weight_mode"])
            budgets.add(int(row["budget"]))
    return fitness, methods, sorted(budgets), modes


def cd_for(k: int, n: int) -> float:
    return float(studentized_range.ppf(1 - ALPHA, k, 1e6)) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))


def analyse(fitness, methods, budgets, mode, seeds, by_seed: bool):
    """(p, cd, mean ranks by method, number of blocks)."""
    if by_seed:
        blocks = []
        for seed in seeds:
            per_budget = [rankdata([-fitness[(mode, b, m, seed)] for m in methods]) for b in budgets]
            blocks.append(list(np.mean(per_budget, axis=0)))
        # Friedman ranks the averaged ranks again within each seed; lower average rank = better.
        ranks = np.array([rankdata(block) for block in blocks])
        statistic, p = friedmanchisquare(*[[-v for v in col] for col in zip(*blocks)])
    else:
        blocks = [[fitness[(mode, b, m, seed)] for m in methods] for b in budgets for seed in seeds]
        ranks = np.array([rankdata([-v for v in block]) for block in blocks])
        statistic, p = friedmanchisquare(*list(zip(*blocks)))
    means = dict(zip(methods, ranks.mean(axis=0)))
    return p, cd_for(len(methods), len(blocks)), means, len(blocks)


def tracked_ranks(stem: str) -> dict[str, dict[str, float]]:
    """Mean ranks per mode as printed in results/<stem>_statistics.log."""
    out, mode = {}, None
    for line in (RESULTS / f"{stem}_statistics.log").read_text(encoding="utf-8").splitlines():
        found = re.match(r"=== (\w+) ===", line)
        if found:
            mode = found.group(1)
            out[mode] = {}
            continue
        found = re.match(r"\s{4}(\w+)\s+([0-9.]+)$", line)
        if found and mode is not None and "rank" not in out.get(mode, {}):
            out[mode].setdefault(found.group(1), float(found.group(2)))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results-invariant" / "friedman_seed_blocks.csv")
    args = ap.parse_args(argv)

    rows, compared, mismatches = [], 0, []
    ranks_of = {"budget_x_seed": {}, "seed": {}}
    for collection, stem in STATS_LOG.items():
        fitness, methods, budgets, modes = load(RESULTS / f"weight_mode_runs_{stem}.csv")
        seeds = sorted({key[3] for key in fitness})
        tracked = tracked_ranks(stem)
        for mode in modes:
            for label, by_seed in (("budget_x_seed", False), ("seed", True)):
                p, cd, means, n = analyse(fitness, methods, budgets, mode, seeds, by_seed)
                order = sorted(means, key=means.get)
                rows.append({"dataset": collection, "mode": mode, "blocks": label, "n_blocks": n,
                             "p": f"{p:.3g}", "cd": f"{cd:.3f}", "best": order[0],
                             "second": order[1], "gap_top2": f"{means[order[1]] - means[order[0]]:.3f}",
                             "top_separated": int(means[order[1]] - means[order[0]] > cd),
                             **{f"rank_{m}": f"{means[m]:.3f}" for m in sorted(methods)}})
                ranks_of[label][(collection, mode)] = (means, cd)
                if not by_seed:
                    for m in methods:
                        compared += 1
                        if f"{means[m]:.3f}" != f"{tracked[mode][m]:.3f}":
                            mismatches.append((collection, mode, m, means[m], tracked[mode][m]))

    print(f"[reproduced] {compared} mean ranks of the 80-block analysis compared with the tracked "
          f"statistics logs, {len(mismatches)} mismatch(es)")
    if compared == 0 or mismatches:
        for item in mismatches[:10]:
            print(f"  [mismatch] {item}")
        print("[refused] the current analysis is not reproduced; nothing written")
        return 1

    reversals = []
    for label, table in ranks_of.items():
        by_mode = defaultdict(list)
        for (collection, mode), value in table.items():
            by_mode[mode].append((collection, *value))
        for mode, entries in by_mode.items():
            methods = sorted(entries[0][1])
            for a, b in combinations(methods, 2):
                for (c1, r1, cd1), (c2, r2, cd2) in combinations(entries, 2):
                    d1, d2 = r1[b] - r1[a], r2[b] - r2[a]
                    if (d1 > cd1 and -d2 > cd2) or (-d1 > cd1 and d2 > cd2):
                        reversals.append({"blocks": label, "mode": mode, "pair": f"{a}-{b}",
                                          "dataset_1": c1, "gap_1": f"{d1:+.3f}", "cd_1": f"{cd1:.3f}",
                                          "dataset_2": c2, "gap_2": f"{d2:+.3f}", "cd_2": f"{cd2:.3f}"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    rev_path = args.out.with_name(args.out.stem + "_reversals.csv")
    with rev_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["blocks", "mode", "pair", "dataset_1", "gap_1", "cd_1", "dataset_2", "gap_2", "cd_2"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(reversals)

    for label in ("budget_x_seed", "seed"):
        kept = [r for r in rows if r["blocks"] == label]
        largest_p = max(float(r["p"]) for r in kept)
        separated = sum(r["top_separated"] for r in kept)
        n_rev = sum(1 for r in reversals if r["blocks"] == label)
        print(f"\n[{label}] {len(kept)} dataset-mode combinations; blocks {kept[0]['n_blocks']}, "
              f"CD {kept[0]['cd']}; largest p {largest_p:.3g}; best ahead of second by more than "
              f"the CD in {separated} of {len(kept)}; {n_rev} reversing pair-dataset combination(s)")
        for r in kept:
            print(f"  {r['dataset']:<18} {r['mode']:<10} p={r['p']:<9} best {r['best']:<12} "
                  f"second {r['second']:<12} gap {r['gap_top2']} {'> CD' if r['top_separated'] else ''}")
        for r in (x for x in reversals if x["blocks"] == label):
            print(f"  reversal {r['mode']:<10} {r['pair']:<24} {r['dataset_1']} {r['gap_1']} / "
                  f"{r['dataset_2']} {r['gap_2']}")
    print(f"[done] {args.out.as_posix()} and {rev_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
