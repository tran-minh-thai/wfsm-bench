"""Can anything computable from the race alone say when the selector will lose?

This exists because a claim failed. The selector's regret bound splits into a term that
vanishes with more race seeds and a term that does not, and the second was described as
computable from the race itself -- so a practitioner would know, before paying the full
budget, which cells the selector handles badly. That is false. Both the spread D and the
regret a perfect race still takes are defined through the dear-budget quality of EVERY
method, and obtaining those means paying the full budget for all of them, which is the work
the selector exists to avoid.

So the question becomes what a race CAN see. It sees one number per method: the mean at the
cheap budget over the race seeds. Two signals follow from that and nothing else:

    gap_top2   how far the leader is ahead of the runner-up, relative to the leader
    spread     how far the leader is ahead of the last, relative to the leader

Both are reported against realised regret, with the correlation and with the behaviour of
the cells where the leader is nearly tied. Counts and correlations only, no timings, so the
artefact is machine-independent.

    python scripts/measure_race_only_signals.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "results-invariant" / "race_only_signals.csv"

# A leader this close to the runner-up is treated as tied at the cheap budget.
TIE = 0.001


def load(patterns: list[str]) -> tuple[dict, list[str]]:
    """The cells, and the files they came from.

    The file list is returned rather than left implicit because a glob resolves against
    whatever a given machine happens to hold. Recording the inputs in the artefact is the
    only way a later reader can tell which runs produced it -- and the only way a check can
    tell whether they are all in the repository.
    """
    cells: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    used: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            used.append(Path(path).resolve().relative_to(ROOT).as_posix())
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (row["dataset"], row["weight_mode"], row["max_size"],
                           row["min_support"])
                    cells[key][int(row["budget"])][row["method"]][int(row["seed"])] = \
                        float(row["fitness"])
    return cells, used


def mean_on(table: dict, method: str, seeds: list[int]) -> float | None:
    values = [table[method][s] for s in seeds if s in table.get(method, {})]
    return sum(values) / len(values) if values else None


def correlation(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else float("nan")


def rows_for(cells: dict) -> list[dict]:
    out = []
    for key, by_budget in cells.items():
        for cheap, dear in itertools.combinations(sorted(by_budget), 2):
            methods = sorted(set(by_budget[cheap]) & set(by_budget[dear]))
            if len(methods) < 3:
                continue
            seeds = sorted(set.intersection(
                *[set(by_budget[cheap][m]) for m in methods],
                *[set(by_budget[dear][m]) for m in methods]))
            race = [s for s in seeds if s % 2 == 0]
            pay = [s for s in seeds if s % 2 == 1]
            if len(race) < 4 or len(pay) < 4:
                continue
            raced = {m: mean_on(by_budget[cheap], m, race) for m in methods}
            paid = {m: mean_on(by_budget[dear], m, pay) for m in methods}
            if None in raced.values() or None in paid.values():
                continue
            best = max(paid.values())
            if best <= 0:
                continue
            ordered = sorted(raced.values(), reverse=True)
            leader = ordered[0]
            pick = max(raced, key=raced.get)
            out.append({
                "dataset": key[0], "weight_mode": key[1], "max_size": key[2],
                "min_support": key[3], "race_budget": cheap, "pay_budget": dear,
                "methods": len(methods), "race_seeds": len(race), "pay_seeds": len(pay),
                "gap_top2": round((leader - ordered[1]) / leader, 6) if leader else "",
                "spread": round((leader - ordered[-1]) / leader, 6) if leader else "",
                "regret_pct": round(100.0 * (best - paid[pick]) / best, 6),
            })
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*",
                    default=[str(RESULTS / "weight_mode_runs_*.csv")])
    args = ap.parse_args(argv)

    cells, used = load(args.runs)
    records = rows_for(cells)
    if not records:
        raise SystemExit("[refused] no cell has enough methods and seeds; a correlation over "
                         "nothing is not a measurement")

    for record in records:
        record["sources"] = " ".join(used)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    regret = [r["regret_pct"] for r in records]
    print(f"[scope] {len(records)} cell(s)")
    for signal in ("gap_top2", "spread"):
        print(f"  correlation(regret, {signal:<9}) = "
              f"{correlation(regret, [r[signal] for r in records]):+.3f}")
    tied = [r for r in records if r["gap_top2"] <= TIE]
    clear = [r for r in records if r["gap_top2"] > TIE]
    for label, group in (("leader tied at the cheap budget", tied),
                         ("leader clear at the cheap budget", clear)):
        if group:
            values = [r["regret_pct"] for r in group]
            print(f"  {label:<34} n={len(group):<3} mean {statistics.mean(values):6.3f}%  "
                  f"worst {max(values):6.3f}%")
    print(f"\n[ok] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
