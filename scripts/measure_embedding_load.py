"""How many embeddings one pattern forces into memory, per collection.

Peak memory in this pipeline is not driven by the number of patterns. Every pattern is
evaluated, scored and discarded, but while it is being evaluated the matcher holds ONE list
of matched edge weights per embedding, for every host that contains it. So the peak is set by
the single worst pattern, and the worst pattern is set by how dense the hosts are -- a dense
host admits enormously many monomorphisms of the same small pattern.

Counts only, no timings, so the numbers do not depend on the machine.

The implementation detail that makes this the right quantity: the matcher enumerates every
monomorphism and de-duplicates AFTERWARDS, so both the work and the memory follow the
monomorphism count, not the count of distinct embeddings that survives.

    python scripts/measure_embedding_load.py
    python scripts/measure_embedding_load.py --cells Cuneiform:60:4 BZR_MD:30:3
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.baselines import mine_exhaustive          # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params  # noqa: E402
from wfsmbench.io import load_dataset                    # noqa: E402
from wfsmbench.match import match_database               # noqa: E402
from wfsmbench.preprocess import build_statistics        # noqa: E402

OUT = ROOT / "results-invariant" / "embedding_load.csv"
DEFAULT = ["Cuneiform:60:3", "Cuneiform:60:4", "BZR_bond:60:4", "BZR_MD:30:3"]


def measure(dataset: str, limit: int, bound: int, min_support: int) -> dict:
    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs[:limit]
    stats = build_statistics(subset)
    params = Params(min_support=min_support, max_nodes=bound, max_edges=bound,
                    size_reference=STUDY_SIZE_REFERENCE)
    pool = mine_exhaustive(subset, stats, params).pool
    counts = [sum(len(v) for v in match_database(c.particle.pattern, subset).values())
              for c in pool]
    if not counts:
        raise SystemExit(f"[refused] {dataset} bound={bound} yields an empty pool; a mean "
                         f"over nothing is not a measurement")
    return {
        "dataset": dataset, "graphs": len(subset), "size_bound": bound,
        "min_support": min_support, "pool": len(pool),
        "edges_mean": round(statistics.mean(g.num_edges() for g in subset), 3),
        "embeddings_mean": round(statistics.mean(counts), 2),
        "embeddings_max": max(counts),
        "embeddings_total": sum(counts),
        # The peak is what has to fit, and how far it sits above the average is the part a
        # mean hides: a collection can look ordinary and still hold one ruinous pattern.
        "peak_over_mean": round(max(counts) / statistics.mean(counts), 2),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=DEFAULT,
                    help="dataset:graphs:size_bound, e.g. Cuneiform:60:4")
    ap.add_argument("--support", type=int, default=2)
    args = ap.parse_args(argv)

    records = []
    for cell in args.cells:
        dataset, limit, bound = cell.split(":")
        records.append(measure(dataset, int(limit), int(bound), args.support))

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"{'collection':<14}{'n':>5}{'bound':>7}{'pool':>7}{'edges/host':>12}"
          f"{'emb/pattern':>13}{'worst':>8}{'worst/mean':>12}")
    for r in records:
        print(f"{r['dataset']:<14}{r['graphs']:>5}{r['size_bound']:>7}{r['pool']:>7}"
              f"{r['edges_mean']:>12.1f}{r['embeddings_mean']:>13.1f}"
              f"{r['embeddings_max']:>8}{r['peak_over_mean']:>11.1f}x")
    print(f"\n[ok] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
