"""Compare the exhaustive (seed-free) columns of a new landscape run with the tracked ones.

Exhaustive enumeration draws no random numbers, so a run of scripts/run_weight_mode.py on the
same data and commit reproduces the pool size, the optimum and the plateau counts of every
cell exactly, on any machine. This checks that for the rows a new run wrote under --suffix,
against the tracked results/weight_mode_landscape*.csv files, and says how many rows it could
compare: zero comparable rows is reported as a failure, not as agreement.

    python scripts/run_weight_mode.py --preset quick --modes sum --seeds 3 --workers 4 --suffix _smoke
    python scripts/compare_landscape.py --suffix _smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = ("dataset", "graph_limit", "max_size", "min_support", "weight_mode")
EXACT = ("pool_size", "optimum_fitness", "optimum_multiplicity", "optimum_nodes",
         "optimum_edges", "optimum_support", "local_optima", "fitness_levels")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", required=True, help="the --suffix the new run was given")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    new_file = args.results / f"weight_mode_landscape{args.suffix}.csv"
    if not new_file.exists():
        print(f"[refused] {new_file} does not exist; run the new landscape first")
        return 2
    tracked = {}
    for path in sorted(args.results.glob("weight_mode_landscape*.csv")):
        if path == new_file:
            continue
        for row in rows(path):
            tracked.setdefault(tuple(row.get(k) for k in KEY), (path.name, row))
    compared, differing = 0, []
    new_rows = rows(new_file)
    for row in new_rows:
        match = tracked.get(tuple(row.get(k) for k in KEY))
        if match is None:
            continue
        compared += 1
        name, old = match
        changed = [(c, row[c], old[c]) for c in EXACT if c in row and c in old and row[c] != old[c]]
        if changed:
            differing.append((tuple(row[k] for k in KEY), name, changed))
    for key, name, changed in differing:
        print(f"[differs] {key} against {name}: {changed}")
    print(f"[compared] {compared} of {len(new_rows)} new row(s) have a tracked counterpart; "
          f"{len(differing)} differ in {', '.join(EXACT)}")
    if compared == 0:
        print("[fail] nothing was compared, so nothing was confirmed")
        return 1
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
