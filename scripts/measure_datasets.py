"""Profile every collection on the quantities this problem actually turns on.

Counts and data properties only -- no timings -- so the numbers do not depend on the machine
and the artefact goes to results-invariant/.

WHY THESE COLUMNS. Three of them decide results in this work rather than merely describing
the data:

  parallel edges   The manuscript once claimed a multigraph model. It is wrong, and the
                   claim came from the project's own name. This is measured from the RAW
                   JSON, where edges are a list, because the in-memory Graph keys edges by
                   an unordered pair and therefore CANNOT hold a parallel edge -- counting
                   them after loading would report zero no matter what the file contained.
                   Raw and loaded edge counts are both reported: if they ever disagree, the
                   loader is discarding edges.

  w_max / w_mean   The slack term of any front-membership bound charges the largest edge
                   weight in the collection for each edge an extension may still add, while
                   the true optimum grows at the mean. That ratio is the factor such a bound
                   must close before it can prune anything.

  self loops       A loop is an edge whose two endpoints coincide, which several definitions
                   here quietly assume away.

    python scripts/measure_datasets.py
    python scripts/measure_datasets.py --collections Cuneiform QM9_bond
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.io import load_dataset          # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results-invariant"


def profile(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    graphs = raw["graphs"]

    raw_edges = parallel = loops = unweighted = 0
    weights: list[float] = []
    node_labels: Counter[int] = Counter()
    edge_labels: Counter[int] = Counter()
    nodes_per_graph, edges_per_graph = [], []

    for graph in graphs:
        seen: Counter[tuple[int, int]] = Counter()
        for edge in graph["edges"]:
            raw_edges += 1
            src, dst = edge["src"], edge["dst"]
            if src == dst:
                loops += 1
            key = (min(src, dst), max(src, dst))
            seen[key] += 1
            edge_labels[edge.get("label")] += 1
            weight = edge.get("weight")
            if weight is None:
                unweighted += 1
            else:
                weights.append(float(weight))
        # An unordered endpoint pair occurring more than once is a parallel edge. Counted
        # per graph, since the same pair in two graphs is two ordinary edges.
        parallel += sum(count - 1 for count in seen.values() if count > 1)
        for vertex in graph["vertices"]:
            node_labels[vertex["label"]] += 1
        nodes_per_graph.append(len(graph["vertices"]))
        edges_per_graph.append(len(graph["edges"]))

    # The loader keys edges by an unordered pair, so a disagreement here means edges were
    # silently dropped on load -- exactly what a parallel edge would do.
    loaded, _ = load_dataset(path)
    loaded_edges = sum(g.num_edges() for g in loaded)

    mean_weight = statistics.mean(weights) if weights else 0.0
    return {
        "dataset": path.stem,
        "graphs": len(graphs),
        "nodes_mean": round(statistics.mean(nodes_per_graph), 4) if nodes_per_graph else "",
        "edges_mean": round(statistics.mean(edges_per_graph), 4) if edges_per_graph else "",
        "edges_total_raw": raw_edges,
        "edges_total_loaded": loaded_edges,
        "edges_lost_on_load": raw_edges - loaded_edges,
        "parallel_edges": parallel,
        "self_loops": loops,
        "node_labels": len(node_labels),
        "edge_labels": len(edge_labels),
        "edges_unweighted": unweighted,
        "weight_min": round(min(weights), 6) if weights else "",
        "weight_mean": round(mean_weight, 6) if weights else "",
        "weight_max": round(max(weights), 6) if weights else "",
        "weight_max_over_mean": round(max(weights) / mean_weight, 4) if mean_weight else "",
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collections", nargs="*",
                    help="dataset JSON stems in data/; default is every one of them")
    args = ap.parse_args(argv)

    paths = ([DATA / f"{name}.json" for name in args.collections] if args.collections
             else sorted(DATA.glob("*.json")))
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"[refused] no such collection: {', '.join(str(p) for p in missing)}")

    records = [profile(path) for path in paths]
    OUT.mkdir(exist_ok=True)
    out = OUT / "dataset_profile.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    width = max(len(r["dataset"]) for r in records)
    print(f"{'collection':<{width}}  {'graphs':>7}  {'nodes':>7}  {'edges':>7}  "
          f"{'parallel':>8}  {'loops':>6}  {'lost':>5}  {'wmax/mean':>9}")
    for r in records:
        print(f"{r['dataset']:<{width}}  {r['graphs']:>7}  {r['nodes_mean']:>7.2f}  "
              f"{r['edges_mean']:>7.2f}  {r['parallel_edges']:>8}  {r['self_loops']:>6}  "
              f"{r['edges_lost_on_load']:>5}  "
              f"{r['weight_max_over_mean'] if r['weight_max_over_mean'] else 0:>9.2f}")
    total = sum(r["edges_total_raw"] for r in records)
    print(f"\n[ok] {len(records)} collection(s), {total} raw edge(s), "
          f"{sum(r['parallel_edges'] for r in records)} parallel, "
          f"{sum(r['self_loops'] for r in records)} self-loop(s) -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
