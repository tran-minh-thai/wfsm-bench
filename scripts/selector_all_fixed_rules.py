"""Regret of every fixed rule, scored exactly as the selector table scores always-ACO and always-ILS.

The selector table compares racing with two fixed rules only. A reader can fairly ask whether
some other method, always chosen, would lose less than both. This answers it with the same
cells, the same disjoint seed halves and the same regret function as
results/selector_regret_allpairs.csv: it imports run_selector rather than restating it.

Before writing anything it re-derives the three columns that artefact already holds (race,
always-ACO, always-ILS) and refuses unless every value matches to the printed precision. A
reproduction that has drifted from the table it extends would make the new columns
incomparable with the old ones.

    python scripts/selector_all_fixed_rules.py
    python scripts/selector_all_fixed_rules.py --out results-invariant/selector_fixed_rules.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_selector as rs  # noqa: E402

TRACKED = ROOT / "results" / "selector_regret_allpairs.csv"
GROUPS = {"all": lambda r: True,
          "sharp": lambda r: r["top"] == "sharp",
          "plateau": lambda r: r["top"] == "plateau"}


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results-invariant" / "selector_fixed_rules.csv")
    args = ap.parse_args(argv)

    paths = sorted({p for pattern in rs.ALL_PAIRS_RUNS for p in glob.glob(str(ROOT / pattern))})
    if not paths:
        print("[fail] none of run_selector.ALL_PAIRS_RUNS matched; nothing to score")
        return 1
    cells = rs.load(paths)
    rows = rs.evaluate(cells, split=True, all_pairs=True)
    if not rows:
        print("[fail] no cell could be scored")
        return 1

    methods = sorted({m for key in cells for budget in cells[key].values() for m in budget})
    with TRACKED.open(encoding="utf-8", newline="") as handle:
        tracked = {(r["dataset"], r["weight_mode"], int(r["max_size"]), int(r["min_support"]),
                    int(r["race_budget"]), int(r["pay_budget"])): r
                   for r in csv.DictReader(handle) if r["seed_split"] == "disjoint"}

    out_rows, compared, mismatches = [], 0, []
    for row in rows:
        key = (*row["cell"], row["cheap"], row["dear"])
        dear = cells[row["cell"]][row["dear"]]
        scored_methods = sorted(set(cells[row["cell"]][row["cheap"]]) & set(dear))
        seeds = rs.scoring_seeds(cells, row["cell"], row["cheap"], row["dear"], scored_methods)
        pay_seeds = [s for s in seeds if s % 2 == 1]
        fixed = {m: rs.regret(m, dear, pay_seeds, scored_methods) for m in methods}
        # Seeds actually present at the paying budget for every method. Since run_selector
        # draws both halves from seeds present at both budgets, this equals pay_seeds there;
        # it is recorded so that the equality is checked rather than assumed.
        present = len({s for s in pay_seeds if all(s in dear.get(m, {}) for m in scored_methods)})
        if present != row["pay_seeds"]:
            print(f"[refused] {key}: {present} paying seed(s) present against "
                  f"{row['pay_seeds']} counted by run_selector; nothing written")
            return 1
        old = tracked.get(key)
        if old is not None:
            compared += 1
            for column, value in (("regret_race", row["race"]), ("regret_aco", fixed.get("aco")),
                                  ("regret_ils", fixed.get("local_search")),
                                  ("regret_pso", fixed.get("pso"))):
                if fmt(value) != old[column]:
                    mismatches.append((key, column, fmt(value), old[column]))
        out_rows.append({"dataset": key[0], "weight_mode": key[1], "max_size": key[2],
                         "min_support": key[3], "race_budget": key[4], "pay_budget": key[5],
                         "race_seeds": row["race_seeds"], "pay_seeds_present": present,
                         "top": row["top"], "picked": row["pick"],
                         "regret_race": fmt(row["race"]),
                         **{f"regret_{m}": fmt(fixed[m]) for m in methods}})

    print(f"[reproduced] {compared} of {len(tracked)} tracked rows compared, "
          f"{len(out_rows)} rows scored, {len(mismatches)} mismatch(es)")
    if compared != len(tracked) or mismatches:
        for item in mismatches[:10]:
            print(f"  [mismatch] {item}")
        print("[refused] the reproduction does not match the tracked table; nothing written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    strategies = ["race", *methods]
    print(f"\n{'group':<9}{'cells':>6}  " + "".join(f"{s:>14}" for s in strategies))
    for name, keep in GROUPS.items():
        kept = [r for r in out_rows if keep(r)]
        line = f"{name:<9}{len(kept):>6}  "
        for s in strategies:
            values = [float(r[f"regret_{s}"]) for r in kept if r[f"regret_{s}"]]
            line += f"{statistics.mean(values):>8.2f}/{max(values):>5.2f}" if values else f"{'-':>14}"
        print(line)
    print("  (mean / worst regret in percent, disjoint seeds)")
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\n[done] {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
