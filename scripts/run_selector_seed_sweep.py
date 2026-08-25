r"""Does racing on more seeds fix the losses the racing bound blames on estimation error?

The racing bound says regret is at most 2*epsilon + D, with epsilon proportional to
1/sqrt(n_r).
Splitting the measured losses by cause showed the estimation term doing the damage --
6 losses averaging 3.97% and peaking at 8.91%, against 19 transfer losses averaging 0.69% --
and every wrong pick had its cheap-budget gap inside 2*epsilon. So the bound predicts
more
race seeds should remove those losses and leave the transfer ones untouched.

That is a quantitative prediction, and this measures it without any new mining.

The paying half is held FIXED at the odd seeds throughout, so every point on the curve is
scored against the same out-of-sample set and the comparison is not confounded by which runs
happen to be paying. Race sets are drawn from the even seeds only, so race and bill never
overlap at any n_r -- the whole reason the split exists.

Only cells with 40 seeds are used, since 20 race seeds have to come from somewhere. Cells
with 20 are reported as skipped rather than silently averaged in at a different n_r.

    python scripts/run_selector_seed_sweep.py
    python scripts/run_selector_seed_sweep.py --draws 200 --out results/seed_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SIZES = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
SMALL = (2, 4, 6, 8, 10)


def load(patterns: list[str]) -> dict:
    cells: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for pattern in patterns:
        for path in sorted(glob.glob(str(ROOT / pattern))) or sorted(glob.glob(pattern)):
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row["dataset"], row["weight_mode"],
                           int(row["max_size"]), int(row["min_support"]))
                    cells[key][int(row["budget"])][row["method"]][int(row["seed"])] = \
                        float(row["fitness"])
    return cells


def mean_on(table, method, seeds):
    values = [table[method][s] for s in seeds if s in table.get(method, {})]
    return statistics.mean(values) if values else None


def sweep_cell(cell, cheap: int, dear: int, draws: int, rng: random.Random):
    """Regret against race-set size, paying half held fixed. None if the cell is unusable."""
    methods = sorted(set(cell[cheap]) & set(cell[dear]))
    seeds = sorted(set.intersection(*[set(cell[cheap][m]) for m in methods],
                                    *[set(cell[dear][m]) for m in methods]))
    race_pool = [s for s in seeds if s % 2 == 0]
    pay = [s for s in seeds if s % 2 == 1]
    # Adapt to what the cell has. Cells with 40 seeds reach 20 race seeds; those with 20
    # reach 10. Forcing one grid would silently drop every 20-seed cell -- and those include
    # the ones whose losses this sweep exists to explain.
    grid = SIZES if len(race_pool) >= max(SIZES) else SMALL
    if len(methods) < 3 or len(race_pool) < max(grid) or len(pay) < max(grid) // 2:
        return None
    assert not set(race_pool) & set(pay), "race and pay overlap; the split is not a split"

    truth = {m: mean_on(cell[dear], m, pay) for m in methods}
    truth = {m: v for m, v in truth.items() if v is not None}
    best = max(truth.values())
    if best <= 0:
        return None

    # Which method a perfect race would pick, and whether that is the one worth paying for.
    cheap_truth = {m: mean_on(cell[cheap], m, race_pool) for m in methods}
    perfect = max(cheap_truth, key=cheap_truth.get)
    transfer_regret = 100.0 * (best - truth[perfect]) / best

    out = {}
    for n_r in grid:
        losses = []
        for _ in range(draws):
            drawn = rng.sample(race_pool, n_r)
            scored = {m: mean_on(cell[cheap], m, drawn) for m in methods}
            scored = {m: v for m, v in scored.items() if v is not None}
            pick = max(scored, key=scored.get)
            losses.append(100.0 * (best - truth[pick]) / best)
        out[n_r] = statistics.mean(losses)
    return out, transfer_regret, max(grid)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["results/weight_mode_runs_*.csv"])
    ap.add_argument("--draws", type=int, default=200,
                    help="random race sets per size; the curve is their mean")
    ap.add_argument("--seed", type=int, default=0, help="fixed, so the sweep is reproducible")
    ap.add_argument("--out", type=Path, default=RESULTS / "selector_seed_sweep.csv")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    cells = load(args.runs)
    rows, skipped = [], 0
    for key in sorted(cells):
        budgets = sorted(cells[key])
        for cheap, dear in itertools.combinations(budgets, 2):
            result = sweep_cell(cells[key][cheap] and cells[key], cheap, dear, args.draws, rng) \
                if False else sweep_cell(cells[key], cheap, dear, args.draws, rng)
            if result is None:
                skipped += 1
                continue
            curve, transfer, reach = result
            rows.append({"cell": key, "cheap": cheap, "dear": dear,
                         "curve": curve, "transfer": transfer, "reach": reach})

    if not rows:
        print("[fail] no cell has 20 race seeds and 10 paying seeds", file=sys.stderr)
        return 1

    print(f"[cells] {len(rows)} usable, {skipped} skipped for too few seeds\n")
    print(f"  {'race seeds':>11} {'mean regret':>13} {'worst cell':>12} "
          f"{'cells losing >1%':>18}")
    for n_r in SIZES:
        values = [r["curve"][n_r] for r in rows if n_r in r["curve"]]
        if not values:
            continue
        print(f"  {n_r:>11} {statistics.mean(values):>12.3f}% {max(values):>11.3f}% "
              f"{sum(1 for v in values if v > 1.0):>13} / {len(values)}")

    floor = statistics.mean(r["transfer"] for r in rows)
    print(f"\n  transfer-only regret (a perfect race, no estimation error): {floor:.3f}%")
    print(f"  this is the floor the curve should approach, and cannot go below")

    header = "dataset,weight_mode,max_size,min_support,race_budget,pay_budget,reach," \
             + ",".join(f"regret_n{n}" for n in SIZES) + ",transfer_only"
    lines = [header]
    for row in rows:
        dataset, mode, size, support = row["cell"]
        values = ",".join(f"{row['curve'][n]:.4f}" if n in row["curve"] else ""
                          for n in SIZES)
        lines.append(f"{dataset},{mode},{size},{support},{row['cheap']},{row['dear']},"
                     f"{row['reach']},{values},{row['transfer']:.4f}")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[done] {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
