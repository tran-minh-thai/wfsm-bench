"""Measure whether an instance-aware upper bound could ever beat the global one.

This decides, before any miner is written, whether localising the slack term of a
front-membership bound can pay. It reports counts and data properties only -- no timings --
so the numbers do not depend on the machine, and the artefact goes to results-invariant/.

THE QUESTION. A sound bound on the objective of every extension of a pattern with k edges,
under a size bound K, must allow for d = K - k further edges. The global form charges each
of them the largest edge weight anywhere in the collection:

    slack_global = d * w_max

An instance-aware form may charge only what the host graph holding the embedding can
actually supply. Two localisations stay sound. The host graph itself gives

    slack_host = max over hosts of (sum of the d largest edge weights in that host)

and the ball of radius d around the embedding gives something between the two -- edges added
beyond the first can attach to newly added vertices, so nothing tighter than the ball is
sound.

WHAT DECIDES IT. If a ball of radius d already covers the whole host, the ball bound IS the
host bound, and the host bound is all the localisation available. So the deciding quantity is
the ball coverage: the mean share of a host's nodes lying within d hops of a starting node.
Near 1 means there is nothing to localise, and no implementation changes that.

    python scripts/measure_bound_locality.py
    python scripts/measure_bound_locality.py --depth 3 --limit 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.core.graph import Graph          # noqa: E402
from wfsmbench.io import load_dataset           # noqa: E402
from wfsmbench.preprocess import build_statistics  # noqa: E402

OUT = ROOT / "results-invariant"
DEFAULT_COLLECTIONS = ["Cuneiform", "QM9_bond", "BZR_bond", "alchemy_full_bond"]


def top_weight_sum(graph: Graph, depth: int) -> float:
    """The most an instance-aware bound may charge for `depth` further edges in this host."""
    weights = sorted((w for (_lbl, w) in graph.edges.values() if w is not None), reverse=True)
    return float(sum(weights[:depth]))


def ball_share(graph: Graph, start: int, depth: int) -> float:
    """Share of the host's nodes within `depth` hops of `start`; 1.0 means no localisation."""
    total = graph.num_nodes()
    if total == 0:
        return 0.0
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist == depth:
            continue
        for other in graph.neighbors(node):
            if other not in seen:
                seen.add(other)
                queue.append((other, dist + 1))
    return len(seen) / total


def measure(dataset: str, limit: int, depth: int) -> dict:
    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs[:limit]
    stats = build_statistics(subset)

    slack_global = depth * stats.max_edge_weight
    slack_host = max((top_weight_sum(g, depth) for g in subset), default=0.0)
    # Two averagings, because they disagree and only one is conservative. Pooling over all
    # (host, node) pairs weights a host by its size; averaging per host first weights every
    # host equally, which favours the small ones and therefore reports HIGHER coverage. High
    # coverage is the reading that supports the criterion, so the pooled figure is the one to
    # state: it is the reading least favourable to the claim being made from it.
    per_graph = [[ball_share(g, node, depth) for node in g.nodes()] for g in subset]
    shares = [x for row in per_graph for x in row]
    graph_means = [sum(row) / len(row) for row in per_graph if row]
    sizes = [g.num_nodes() for g in subset]

    return {
        "dataset": dataset,
        "graphs": len(subset),
        "depth": depth,
        "w_max": round(stats.max_edge_weight, 6),
        "slack_global": round(slack_global, 6),
        "slack_host_max": round(slack_host, 6),
        # How much tighter the tightest sound localisation is. The bound must close a factor
        # of w_max / mean-weight to fire at all, so this is the number to compare against.
        "gain": round(slack_global / slack_host, 6) if slack_host else "",
        "ball_coverage_pooled": round(sum(shares) / len(shares), 6) if shares else "",
        "ball_coverage_per_graph": (round(sum(graph_means) / len(graph_means), 6)
                                    if graph_means else ""),
        "ball_coverage_min": round(min(shares), 6) if shares else "",
        "host_nodes_mean": round(sum(sizes) / len(sizes), 6) if sizes else "",
        "host_nodes_max": max(sizes, default=0),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=3,
                    help="edges an extension may still add, i.e. the size bound minus k")
    ap.add_argument("--limit", type=int, default=200, help="graphs read per collection")
    ap.add_argument("--collections", nargs="*", default=DEFAULT_COLLECTIONS)
    args = ap.parse_args(argv)

    records = [measure(name, args.limit, args.depth) for name in args.collections]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"bound_locality_d{args.depth}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    width = max(len(r["dataset"]) for r in records)
    print(f"{'collection':<{width}}  {'d*w_max':>9}  {'host top-d':>10}  {'gain':>6}  "
          f"{'ball cover':>10}  {'host nodes':>10}")
    for r in records:
        print(f"{r['dataset']:<{width}}  {r['slack_global']:>9.3f}  {r['slack_host_max']:>10.3f}  "
              f"{r['gain']:>5.2f}x  {100 * r['ball_coverage_pooled']:>9.1f}%  "
              f"{r['host_nodes_mean']:>10.1f}")
    print(f"\n[ok] wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
