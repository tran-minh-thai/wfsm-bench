"""Does the min-max scaling of edge weights change the non-dominated front?

Every collection is stored with its edge weights scaled affinely, w = WEIGHT_MIN +
(WEIGHT_MAX - WEIGHT_MIN) * (d - d_min) / (d_max - d_min), from the raw distance d. Under an
affine map with a non-zero offset the total weight of a k-edge embedding shifts by an amount
proportional to k, so ATW and WSUP are not monotone images of their raw values and the front
over (WSUP, ATW, -size) may differ. This measures by how much, on real cells: it rebuilds the
host graphs with the raw distances recovered from data/raw, enumerates the same pool, and
compares the two fronts pattern by pattern, together with where each selector lands.

The pool itself cannot differ (support and the size bound do not read weights); the script
checks that before comparing anything.

Counts only, no timings.

    python scripts/front_raw_weights.py
    python scripts/front_raw_weights.py --dataset Cuneiform --raw-name Cuneiform --sizes 4 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pareto_size as rps                                     # noqa: E402
from wfsmbench.baselines import mine_exhaustive                   # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params         # noqa: E402
from wfsmbench.core.graph import Graph                            # noqa: E402
from wfsmbench.core.signature import canonical_signature          # noqa: E402
from wfsmbench.io import load_dataset                             # noqa: E402
from wfsmbench.io.tudataset import WEIGHT_MAX, WEIGHT_MIN, _edge_distance, _read_lines  # noqa: E402
from wfsmbench.measure.pareto import bounds_of, normalize, objectives, pareto_front  # noqa: E402
from wfsmbench.preprocess import build_statistics                 # noqa: E402


def raw_range(raw_name: str) -> tuple[float, float]:
    path = ROOT / "data" / "raw" / raw_name / f"{raw_name}_edge_attributes.txt"
    distances = [_edge_distance(line) for line in _read_lines(path)]
    return min(distances), max(distances)


def to_raw(graphs: list[Graph], d_min: float, d_max: float) -> list[Graph]:
    span = d_max - d_min
    out = []
    for graph in graphs:
        edges = {key: (label, d_min + (w - WEIGHT_MIN) * span / (WEIGHT_MAX - WEIGHT_MIN))
                 for key, (label, w) in graph.edges.items()}
        out.append(Graph(node_labels=dict(graph.node_labels), edges=edges,
                         graph_id=graph.graph_id))
    return out


def front_of(pool, params):
    """Front members (one per objective vector, as run_pareto_size picks them) and picks."""
    points = [objectives(c.evaluation, c.particle.pattern) for c in pool]
    on_front = set(pareto_front(points))
    members, seen = [], set()
    for candidate, point in sorted(zip(pool, points), key=lambda cp: cp[1], reverse=True):
        if point in on_front and point not in seen:
            seen.add(point)
            members.append(candidate)
    front = [objectives(c.evaluation, c.particle.pattern) for c in members]
    scaled = normalize(front, bounds_of(front))
    picks = {"ideal": rps.pick_ideal(scaled), "maxmin": rps.pick_maxmin(scaled),
             "weighted": rps.pick_weighted(members, params)}
    # Every pattern whose vector is on the front, not only the representative, so that two
    # fronts are compared as sets of patterns.
    all_on_front = {c.signature for c, p in zip(pool, points) if p in on_front}
    return members, picks, all_on_front


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="Cuneiform")
    ap.add_argument("--raw-name", default="Cuneiform")
    ap.add_argument("--graph-limit", type=int, default=rps.STUDY_GRAPH_LIMIT)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results-invariant" / "front_raw_weights.csv")
    args = ap.parse_args(argv)

    graphs, _ = load_dataset(ROOT / "data" / f"{args.dataset}.json")
    scaled_hosts = graphs[:args.graph_limit]
    d_min, d_max = raw_range(args.raw_name)
    raw_hosts = to_raw(scaled_hosts, d_min, d_max)
    a = (WEIGHT_MAX - WEIGHT_MIN) / (d_max - d_min)
    b = WEIGHT_MIN - a * d_min
    print(f"{args.dataset}: raw distance in [{d_min}, {d_max}]; stored w = {a:.6g} d + {b:.6g}")

    rows = []
    for bound in args.sizes:
        for fraction in rps.SUPPORT_FRACTIONS:
            min_support = max(1, int(round(fraction * len(scaled_hosts))))
            params = Params(min_support=min_support, max_nodes=bound, max_edges=bound,
                            size_reference=STUDY_SIZE_REFERENCE)
            pools = {}
            for name, hosts in (("scaled", scaled_hosts), ("raw", raw_hosts)):
                pools[name] = mine_exhaustive(hosts, build_statistics(hosts), params).pool
            sig = {name: {c.signature for c in pool} for name, pool in pools.items()}
            if sig["scaled"] != sig["raw"]:
                print(f"[refused] K={bound} support={min_support}: the two pools differ, "
                      f"which weights cannot cause; nothing written")
                return 1
            got = {name: front_of(pool, params) for name, pool in pools.items()}
            common = got["scaled"][2] & got["raw"][2]
            row = {"dataset": args.dataset, "graphs": len(scaled_hosts), "size_bound": bound,
                   "min_support": min_support, "pool": len(pools["scaled"]),
                   "front_patterns_scaled": len(got["scaled"][2]),
                   "front_patterns_raw": len(got["raw"][2]),
                   "front_patterns_common": len(common),
                   "front_vectors_scaled": len(got["scaled"][0]),
                   "front_vectors_raw": len(got["raw"][0])}
            for name in ("scaled", "raw"):
                members, picks, _ = got[name]
                edges = [c.particle.pattern.num_edges() for c in members]
                row[f"front_edges_{name}"] = "|".join(str(e) for e in sorted(set(edges)))
                for selector, index in picks.items():
                    row[f"{selector}_edges_{name}"] = edges[index]
                    row[f"{selector}_same_pattern"] = ""
            for selector in ("ideal", "maxmin", "weighted"):
                s_members, s_picks, _ = got["scaled"]
                r_members, r_picks, _ = got["raw"]
                row[f"{selector}_same_pattern"] = int(
                    s_members[s_picks[selector]].signature == r_members[r_picks[selector]].signature)
            print(f"K={bound} support={min_support}: pool {row['pool']}, front patterns "
                  f"scaled {row['front_patterns_scaled']} raw {row['front_patterns_raw']} "
                  f"common {row['front_patterns_common']}; picks (edges scaled/raw) "
                  + ", ".join(f"{s} {row[f'{s}_edges_scaled']}/{row[f'{s}_edges_raw']}"
                              f"{'' if row[f'{s}_same_pattern'] else ' (different pattern)'}"
                              for s in ("ideal", "maxmin", "weighted")), flush=True)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {len(rows)} cell(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
