"""Embedding load of the worst pattern, counted over the database and within a single host.

results-invariant/embedding_load.csv records, for each cell, the largest number of embeddings
one pattern holds at once when the matcher returns them for every host together, which is what
match_database does today: it builds the embeddings of all hosts before the evaluation reduces
them. A matcher that reduced each host before moving to the next would hold only the largest
per-host count. This counts both, on the same cells and the same pools, so the two memory
statements can be compared on numbers rather than on the formula alone.

It re-derives the database-wide maximum and total and refuses unless they equal the tracked
embeddings_max and embeddings_total of the same cell.

Counts only, no timings, so the numbers do not depend on the machine.

    python scripts/embedding_load_per_host.py
    python scripts/embedding_load_per_host.py --cells Cuneiform:60:3 --out /tmp/x.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_embedding_load import DEFAULT, OUT as TRACKED  # noqa: E402
from wfsmbench.baselines import mine_exhaustive              # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params    # noqa: E402
from wfsmbench.io import load_dataset                        # noqa: E402
from wfsmbench.match import match_database                   # noqa: E402
from wfsmbench.preprocess import build_statistics            # noqa: E402


def measure(dataset: str, limit: int, bound: int, min_support: int) -> dict:
    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs[:limit]
    stats = build_statistics(subset)
    params = Params(min_support=min_support, max_nodes=bound, max_edges=bound,
                    size_reference=STUDY_SIZE_REFERENCE)
    pool = mine_exhaustive(subset, stats, params).pool
    if not pool:
        raise SystemExit(f"[refused] {dataset} bound={bound} yields an empty pool")
    database, per_host = [], []
    for candidate in pool:
        counts = [len(v) for v in match_database(candidate.particle.pattern, subset).values()]
        database.append(sum(counts))
        per_host.append(max(counts) if counts else 0)
    worst = max(range(len(pool)), key=lambda i: database[i])
    return {"dataset": dataset, "graphs": len(subset), "size_bound": bound,
            "min_support": min_support, "pool": len(pool),
            "embeddings_max": max(database), "embeddings_total": sum(database),
            "per_host_max": max(per_host),
            "per_host_of_database_worst": per_host[worst],
            "database_over_per_host": round(max(database) / max(per_host), 2)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=DEFAULT, help="dataset:graphs:bound")
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results-invariant" / "embedding_load_per_host.csv")
    args = ap.parse_args(argv)

    with TRACKED.open(encoding="utf-8", newline="") as handle:
        tracked = {(r["dataset"], r["graphs"], r["size_bound"], r["min_support"]): r
                   for r in csv.DictReader(handle)}
    rows, compared = [], 0
    for cell in args.cells:
        dataset, limit, bound = cell.split(":")
        row = measure(dataset, int(limit), int(bound), args.min_support)
        old = tracked.get((dataset, str(row["graphs"]), bound, str(args.min_support)))
        if old is not None:
            compared += 1
            for column in ("embeddings_max", "embeddings_total"):
                if str(row[column]) != old[column]:
                    print(f"[refused] {cell}: {column} {row[column]} differs from the tracked "
                          f"{old[column]}; nothing written")
                    return 1
        print(f"{cell}: pool {row['pool']}, database max {row['embeddings_max']}, "
              f"per-host max {row['per_host_max']} "
              f"(ratio {row['database_over_per_host']})", flush=True)
        rows.append(row)
    print(f"[reproduced] {compared} of {len(rows)} cell(s) matched the tracked embedding load")
    if compared == 0:
        print("[refused] no cell could be compared with the tracked artefact; nothing written")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
