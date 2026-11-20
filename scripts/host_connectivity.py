"""Are the host graphs connected, and how much of the ball coverage does that explain?

The ball coverage of results-invariant/bound_locality_d*.csv divides the nodes reached within d
hops by the number of nodes in the whole host. On a host with several connected components no
ball can reach the other components, so coverage is capped by the share of the host's nodes in
the starting node's component. This reports, per collection and on the same graphs, how many
hosts are connected, how many components they have, and the ball coverage recomputed with the
component as denominator, next to the tracked whole-host value which it re-derives first.

Counts only, no timings, so the numbers do not depend on the machine.

    python scripts/host_connectivity.py
    python scripts/host_connectivity.py --depth 2 --out /tmp/x.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_bound_locality import DEFAULT_COLLECTIONS, ball_share  # noqa: E402
from wfsmbench.io import load_dataset                                # noqa: E402


def components(graph) -> list[set]:
    left, out = set(graph.nodes()), []
    while left:
        start = left.pop()
        seen, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for other in graph.neighbors(node):
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        left -= seen
        out.append(seen)
    return out


def measure(dataset: str, limit: int, depth: int) -> dict:
    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs[:limit]
    counts, whole, within, isolated = [], [], [], 0
    for graph in subset:
        parts = components(graph)
        counts.append(len(parts))
        isolated += sum(1 for part in parts if len(part) == 1)
        size_of = {node: len(part) for part in parts for node in part}
        for node in graph.nodes():
            share = ball_share(graph, node, depth)
            whole.append(share)
            within.append(share * graph.num_nodes() / size_of[node])
    return {"dataset": dataset, "graphs": len(subset), "depth": depth,
            "connected_graphs": sum(1 for c in counts if c == 1),
            "connected_share": round(sum(1 for c in counts if c == 1) / len(counts), 6),
            "components_mean": round(statistics.mean(counts), 4),
            "components_max": max(counts),
            "isolated_nodes": isolated,
            "ball_coverage_pooled": round(sum(whole) / len(whole), 6),
            "ball_coverage_within_component": round(sum(within) / len(within), 6)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results-invariant" / "host_connectivity.csv")
    args = ap.parse_args(argv)

    tracked_path = ROOT / "results-invariant" / f"bound_locality_d{args.depth}.csv"
    with tracked_path.open(encoding="utf-8", newline="") as handle:
        tracked = {r["dataset"]: r for r in csv.DictReader(handle)}
    rows, compared = [], 0
    for name in DEFAULT_COLLECTIONS:
        row = measure(name, args.limit, args.depth)
        old = tracked.get(name)
        if old is not None and old["graphs"] == str(row["graphs"]):
            compared += 1
            if f"{row['ball_coverage_pooled']}" != f"{float(old['ball_coverage_pooled'])}":
                print(f"[refused] {name}: pooled coverage {row['ball_coverage_pooled']} differs "
                      f"from the tracked {old['ball_coverage_pooled']}; nothing written")
                return 1
        print(f"{name}: {row['connected_graphs']}/{row['graphs']} connected, "
              f"components mean {row['components_mean']} max {row['components_max']}, "
              f"coverage whole host {row['ball_coverage_pooled']:.4f}, "
              f"within component {row['ball_coverage_within_component']:.4f}", flush=True)
        rows.append(row)
    print(f"[reproduced] {compared} of {len(rows)} collection(s) matched {tracked_path.name}")
    if compared == 0:
        print("[refused] nothing could be compared with the tracked artefact; nothing written")
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
