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

# always-PSO joined on 2026-11-20: scored over all seven methods it is the strongest fixed rule
# on these cells, and comparing racing with weaker rules only would overstate it.
FIXED = ("aco", "local_search", "pso")

# The stage 13 per-seed files: two budgets each, disjoint seeds for race and bill.
STAGE_13_RUNS = ["results/weight_mode_runs_sel*.csv"]

# The inputs of results/selector_regret_allpairs.csv, the table the selector's regret is
# quoted from. Four budget levels are what give it more than one budget pair per cell, so it
# needs the method arms that ran at four -- QM9 (stage 2), Cuneiform at sizes 4, 5 and 6
# (stages 8, 4 and 7) and the size-6 pruned arm (stage 5) -- beside the stage 13 files.
#
# CORRECTED 2026-09-27. This list was written nowhere. A commit of 2026-08-24 deleted this
# script's main() along with a dead line it meant to remove, so stage 13 ran it as a no-op
# for a month, printing nothing and exiting zero; and the table could not be regenerated,
# because which files had produced it was not recorded. Recovered by search over the tracked
# per-seed files: this set, and only it among the candidates, reproduces the table byte for
# byte (122 rows). weight_mode_runs_macdet.csv is byte-identical to s4loose and would
# serve equally; weight_mode_runs_noprune.csv would not.
ALL_PAIRS_RUNS = ["results/weight_mode_runs_sel*.csv",
                  "results/weight_mode_runs_qm9.csv",
                  "results/weight_mode_runs_s4loose.csv",
                  "results/weight_mode_runs_s5.csv",
                  "results/weight_mode_runs_s6.csv",
                  "results/weight_mode_runs_prune.csv"]


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


def scoring_seeds(cells, key, cheap, dear, methods) -> list[int]:
    """Seeds present at both budgets for every method: the only seeds race and bill may use."""
    if not methods:
        return []
    return sorted(set.intersection(*[set(cells[key][budget][m])
                                     for budget in (cheap, dear) for m in methods]))


def _score(cells, key, cheap, dear, split: bool):
    """One (cell, budget pair), or [] when it cannot be scored."""
    methods = sorted(set(cells[key][cheap]) & set(cells[key][dear]))
    # CORRECTED 2026-11-20. Race and bill are drawn from the seeds present at BOTH budgets for
    # every method. The seeds used to be read from the racing budget alone, and eight cells
    # merge a 40-seed stage-13 file (budgets 60 and 240) with a 20-seed file (all four
    # budgets): at a paying budget of 120 or 180 they raced on 20 seeds but could pay on only
    # the 10 odd seeds that exist there, while the pay_seeds column still said 20. Racing on
    # seeds that the bill cannot see is not the protocol, and it made this table disagree with
    # the transfer column of results/selector_seed_sweep.csv, which already used this rule.
    seeds = scoring_seeds(cells, key, cheap, dear, methods)
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


def summarise(rows, label):
    print(f"\n  == {label} ==")
    print(f"  {'dataset':<20}{'mode':<12}{'picked':<14}{'race':>8}{'aco':>9}{'ils':>9}"
          f"{'race/pay seeds':>16}")
    for row in rows:
        dataset, mode, _size, _support = row["cell"]
        def show(v): return f"{v:>8.2f}%" if v is not None else f"{'-':>9}"
        print(f"  {dataset:<20}{mode:<12}{row['pick']:<14}{show(row['race'])}"
              f"{show(row['aco'])}{show(row['local_search'])}"
              f"{row['race_seeds']:>7}/{row['pay_seeds']:<6}")
    print()
    out = {}
    for name, label2 in (("race", "race"), ("aco", "always-ACO"),
                         ("local_search", "always-ILS"), ("pso", "always-PSO")):
        values = [r[name] for r in rows if r[name] is not None]
        if not values:
            print(f"  {label2:<14} no cell could be compared")
            continue
        out[name] = (statistics.mean(values), max(values), len(values))
        print(f"  {label2:<14} mean regret {statistics.mean(values):>6.2f}%   "
              f"worst {max(values):>6.2f}%   over {len(values)} cells")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="per-seed CSV files; globs are expanded. Default: the stage 13 "
                         "files, or ALL_PAIRS_RUNS with --all-pairs")
    ap.add_argument("--out", type=Path, default=RESULTS / "selector_regret.csv")
    ap.add_argument("--all-pairs", action="store_true",
                    help="score every budget pair, not just cheapest against dearest; this "
                         "is what reaches cells with a sharp top")
    args = ap.parse_args(argv)
    runs = args.runs or (ALL_PAIRS_RUNS if args.all_pairs else STAGE_13_RUNS)

    # Relative patterns resolve against the repository, not the directory the command was
    # typed in, so the same invocation reads the same files from anywhere.
    paths = sorted({p for pattern in runs
                    for p in glob.glob(str(pattern) if Path(pattern).is_absolute()
                                       else str(ROOT / pattern))})
    if not paths:
        print(f"[fail] no per-seed files matched {args.runs}. A selector scored over zero "
              f"cells is not a scored selector.")
        return 1
    print(f"reading {len(paths)} per-seed files")
    cells = load(paths)
    if not cells:
        print("[fail] the files held no rows.")
        return 1

    split_rows = evaluate(cells, split=True, all_pairs=args.all_pairs)
    shared_rows = evaluate(cells, split=False, all_pairs=args.all_pairs)
    if not split_rows:
        print("[fail] no cell had two budgets and enough seeds to split.")
        return 1

    split_summary = summarise(split_rows, "DISJOINT SEEDS -- race on even, pay on odd")
    shared_summary = summarise(shared_rows, "SHARED SEEDS -- as measured before, to size the optimism")

    if "race" in split_summary and "race" in shared_summary:
        honest, optimistic = split_summary["race"][0], shared_summary["race"][0]
        print(f"\n  optimism of the shared-seed figure: {optimistic:.2f}% -> "
              f"{honest:.2f}% ({honest - optimistic:+.2f} percentage points)")
        best_fixed = min((split_summary[f][0] for f in FIXED if f in split_summary),
                         default=None)
        if best_fixed is not None:
            print(f"  race against the best fixed rule, on disjoint seeds: "
                  f"{honest:.2f}% vs {best_fixed:.2f}%  -> "
                  f"{'race still wins' if honest < best_fixed else 'RACE NO LONGER WINS'}")

    header = ("dataset,weight_mode,max_size,min_support,seed_split,race_budget,"
              "pay_budget,picked,race_seeds,pay_seeds,top,tied_at_top,transfers,"
              "estimated_right,regret_race,regret_aco,regret_ils,regret_pso")
    lines = [header]
    for label, rows in (("disjoint", split_rows), ("shared", shared_rows)):
        for row in rows:
            dataset, mode, size, support = row["cell"]
            def num(v): return f"{v:.4f}" if v is not None else ""
            lines.append(f"{dataset},{mode},{size},{support},{label},{row['cheap']},"
                         f"{row['dear']},{row['pick']},{row['race_seeds']},"
                         f"{row['pay_seeds']},{row['top']},{row['tied_at_top']},"
                         f"{row['transfers']},{row['estimated_right']},"
                         f"{num(row['race'])},{num(row['aco'])},"
                         f"{num(row['local_search'])},{num(row['pso'])}")
    # newline="": the same bytes on every system. The default translates to CRLF on Windows,
    # so the measurement machine would rewrite an unchanged table as a changed file.
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\n[done] {shown}")
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake      # Windows: no sleep while the stage runs
    keep_awake()
    raise SystemExit(main(sys.argv[1:]))
