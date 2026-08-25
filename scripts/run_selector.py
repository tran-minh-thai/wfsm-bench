"""Score the racing selector on seeds it did not choose on.

Racing the methods at the cheaper budget and keeping the winner beats every fixed rule.
An early measurement raced and paid on the same seeds, so the race saw exactly the runs
it was later judged on and the number was optimistic by an unknown amount. Unknown is
the problem: a selector reported at one figure when it is really several times that is a
different recommendation. No figure is quoted in this docstring on purpose -- the numbers
live in results/selector_regret_allpairs.csv and in the generated tables, and a figure
copied into a comment is one that will not move when the artefact does.

This splits the seeds. The race sees the even ones, the bill is paid on the odd ones,
and no run appears in both. Regret is the loss against the best method on the paying
seeds, so a selector that picks a method which is merely tied loses nothing, which is
the right accounting -- naming a different winner inside a statistical tie is not an
error, and scoring by "did it name the same method" would call it one.

The shared-seed figure is computed alongside deliberately. The gap between the two is
the size of the optimism, which turns a caveat into a measurement, and it is the number
that says whether the earlier result needs restating or merely annotating.

Three strategies are compared on identical cells:

    race        pick the best mean at the cheap budget on the race seeds
    always-aco  the best fixed rule these collections offer
    always-ils  a fixed rule for the exploiting side, for scale

An oracle scores zero by construction and is not listed; the interesting comparison is
against the best rule someone could actually follow without running anything.

    python scripts/run_selector.py --runs results/weight_mode_runs_sel*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FIXED = ("aco", "local_search")


def load(paths):
    """(dataset, mode, size, support) -> budget -> method -> seed -> fitness."""
    cells: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], row["weight_mode"],
                       int(row["max_size"]), int(row["min_support"]))
                cells[key][int(row["budget"])][row["method"]][int(row["seed"])] = \
                    float(row["fitness"])
    return cells


def mean_over(table, method, seeds):
    values = [table[method][s] for s in seeds if s in table.get(method, {})]
    return statistics.mean(values) if values else None


def regret(pick, pay_table, pay_seeds, methods):
    """Percentage below the best method available, measured on the paying seeds."""
    scores = {m: mean_over(pay_table, m, pay_seeds) for m in methods}
    scores = {m: v for m, v in scores.items() if v is not None}
    if not scores or pick not in scores:
        return None
    best = max(scores.values())
    if best <= 0:
        return None
    return 100.0 * (best - scores[pick]) / best


def budget_pairs(budgets: list[int], all_pairs: bool):
    """Which (race, pay) budget pairs to score.

    The default takes the cheapest and dearest only, which is the selector as proposed. With
    all_pairs every ordered pair is scored, and that is not cosmetic: nine cells carry four
    budget levels, so it turns 11 comparisons into 61 and -- more to the point -- it reaches
    cells with a **sharp top**, where no two methods tie at the paying budget. The plateau
    cells are the ones where the selector cannot lose, so an evaluation made only of them
    cannot say whether the selector is any good.
    """
    if not all_pairs:
        return [(budgets[0], budgets[-1])]
    return [(a, b) for i, a in enumerate(budgets) for b in budgets[i + 1:]]


def evaluate(cells, split: bool, all_pairs: bool = False):
    """One row per (cell, budget pair). `split` sends race and bill to disjoint seed halves."""
    rows = []
    for key in sorted(cells):
        budgets = sorted(cells[key])
        if len(budgets) < 2:
            continue
        for cheap, dear in budget_pairs(budgets, all_pairs):
            rows.extend(_score(cells, key, cheap, dear, split))
    return rows


def _score(cells, key, cheap, dear, split: bool):
    """One (cell, budget pair), or [] when it cannot be scored."""
    methods = sorted(set(cells[key][cheap]) & set(cells[key][dear]))
    seeds = sorted({s for m in cells[key][cheap].values() for s in m})
    if len(seeds) < 4 or len(methods) < 2:
        return []
    race_seeds = [s for s in seeds if s % 2 == 0] if split else seeds
    pay_seeds = [s for s in seeds if s % 2 == 1] if split else seeds
    if not race_seeds or not pay_seeds:
        return []
    if split and set(race_seeds) & set(pay_seeds):
        raise SystemExit("[error] the seed halves overlap; the split is not a split")

    scored = {m: mean_over(cells[key][cheap], m, race_seeds) for m in methods}
    scored = {m: v for m, v in scored.items() if v is not None}
    if not scored:
        return []
    pick = max(scored, key=scored.get)
    # Classify the cell, because the two kinds behave completely differently and an
    # average over both hides it. A "plateau" top has two or more methods tied at the
    # paying budget, so naming any of them costs nothing; a "sharp" top has one winner
    # and the choice is real.
    truth_pay = {m: mean_over(cells[key][dear], m, pay_seeds) for m in methods}
    truth_pay = {m: v for m, v in truth_pay.items() if v is not None}
    best_pay = max(truth_pay.values()) if truth_pay else 0.0
    ties = sum(1 for v in truth_pay.values() if best_pay - v <= 1e-9)

    truth_race = {m: mean_over(cells[key][cheap], m, seeds) for m in methods}
    truth_race = {m: v for m, v in truth_race.items() if v is not None}
    cheap_best = max(truth_race, key=truth_race.get) if truth_race else None
    dear_best = max(truth_pay, key=truth_pay.get) if truth_pay else None

    row = {"cell": key, "cheap": cheap, "dear": dear, "pick": pick,
           "race_seeds": len(race_seeds), "pay_seeds": len(pay_seeds),
           "top": "plateau" if ties > 1 else "sharp", "tied_at_top": ties,
           # which of the two terms of the bound is responsible when it loses
           "transfers": int(cheap_best == dear_best),
           "estimated_right": int(pick == cheap_best),
           "race": regret(pick, cells[key][dear], pay_seeds, methods)}
    for fixed in FIXED:
        row[fixed] = regret(fixed, cells[key][dear], pay_seeds, methods)
    return [row]
