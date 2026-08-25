"""Compare two runs of the same cell and report whether they agree exactly.

The question this answers is not "is one machine faster" -- that is a timing
question and none of the reported numbers are timings. It is whether the result
set is a *function of the commit*: same code, same data, same seeds, same library
versions, therefore same numbers, whatever the machine underneath. If that holds,
a reviewer reproduces the tables on their own hardware and the provenance of any
individual file stops mattering. If it does not hold, every number is bound to
one machine and only a single-machine run of the whole suite is quotable.

The comparison is exact. These are deterministic searches over a fixed pattern
pool with seeded generators, so "close enough" is the wrong tolerance: a drift of
1e-12 means the streams diverged, and a tolerance would hide exactly the thing
worth finding.

Counts are printed for every family compared, because "no disagreements" over
zero comparable rows is an unverified check wearing the clothes of a passed one.
Exits non-zero on any disagreement, on an empty comparison, or on a missing file.

    python scripts/check_determinism.py --a s4loose --b macdet

Use it after re-running a stage on another machine, and after the final tagged
run of the whole suite, to confirm nothing drifted between them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# family -> (key columns, value columns). Values are compared as text, because
# these files are what gets read back and quoted; a difference invisible in the
# text is a difference nobody would ever act on.
FAMILIES = {
    "weight_mode_methods": (
        ["dataset", "max_size", "min_support", "weight_mode", "budget", "method"],
        ["fitness_mean", "fitness_std", "gap_pct", "hits", "hits_fitness_tie",
         "optimum_multiplicity"],
    ),
    "weight_mode_landscape": (
        ["dataset", "max_size", "min_support", "weight_mode"],
        ["pool_size", "frequent_templates", "optimum_fitness", "optimum_multiplicity",
         "optimum_edges", "optimum_support", "optimum_is_interior",
         "spearman_f_edges", "local_optima", "fitness_levels"],
    ),
    "weight_mode_runs": (
        ["dataset", "max_size", "min_support", "weight_mode", "budget", "method", "seed"],
        ["fitness", "found_optimum"],
    ),
}


def read(path: Path, keys: list[str], values: list[str]) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in keys + values if c not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"{path.name} has no column {', '.join(missing)}")
        return {tuple(row[k] for k in keys): tuple(row[v] for v in values)
                for row in reader}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def compare(family: str, a: Path, b: Path) -> tuple[int, int, list[str]]:
    keys, values = FAMILIES[family]
    left, right = read(a, keys, values), read(b, keys, values)
    only_left, only_right = set(left) - set(right), set(right) - set(left)
    shared = sorted(set(left) & set(right))
    notes = []
    for label, extra in (("only in " + a.name, only_left), ("only in " + b.name, only_right)):
        if extra:
            notes.append(f"{len(extra)} rows {label}, e.g. {sorted(extra)[0]}")
    bad = [k for k in shared if left[k] != right[k]]
    for key in bad[:5]:
        differing = [f"{c}: {x} vs {y}"
                     for c, x, y in zip(values, left[key], right[key]) if x != y]
        notes.append(f"{'/'.join(key)} -> {'; '.join(differing)}")
    return len(shared), len(bad), notes


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="suffix of the first run, e.g. s4loose")
    ap.add_argument("--b", required=True, help="suffix of the second run, e.g. macdet")
    ap.add_argument("--families", nargs="+", default=sorted(FAMILIES),
                    choices=sorted(FAMILIES),
                    help="which families to compare. Two runs over different seed counts "
                         "or budget grids share their per-seed rows but not their "
                         "aggregates, so comparing the summary families there would "
                         "report a difference that is arithmetic rather than drift")
    args = ap.parse_args(argv)

    print(f"comparing _{args.a} against _{args.b}\n")
    total_rows = total_bad = families = 0
    broken: list[str] = []
    for family in args.families:
        a = RESULTS / f"{family}_{args.a}.csv"
        b = RESULTS / f"{family}_{args.b}.csv"
        if not a.exists() or not b.exists():
            absent = a.name if not a.exists() else b.name
            print(f"  {family:<22} skipped, no {absent}")
            continue
        families += 1
        try:
            rows, bad, notes = compare(family, a, b)
        except KeyError as error:
            print(f"  {family:<22} BROKEN -- {error}")
            broken.append(family)
            continue
        identical = "byte-identical" if digest(a) == digest(b) else "differing bytes"
        print(f"  {family:<22} {rows:>5} rows compared, {bad} disagree   ({identical})")
        for note in notes:
            print(f"      {note}")
        total_rows += rows
        total_bad += bad

    if broken:
        print(f"\n[fail] {len(broken)} famil{'y' if len(broken) == 1 else 'ies'} could not "
              f"be compared at all: {', '.join(broken)}. A check that skips a family "
              f"silently reports on what it reached, not on what it was asked about.")
        return 1
    if families == 0:
        print("\n[fail] nothing was compared -- no family had both files present.")
        return 1
    if total_rows == 0:
        print("\n[fail] every file pair was empty; this check verified nothing.")
        return 1
    print(f"\n{total_rows} rows across {families} families, {total_bad} disagreements")
    if total_bad:
        print("[fail] the two runs are not the same function of the commit.")
        return 1
    print("[ok] identical -- these numbers depend on the commit, not on the machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
