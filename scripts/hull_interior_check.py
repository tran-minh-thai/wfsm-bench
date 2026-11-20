"""Where can the penalty coefficient put the optimum at an intermediate size?

F(g) = G(g) - gamma * Pen(g), with G the reward term and Pen = eta1*|V| + eta2*|E|. For a fixed
pool, the maximiser of F over all gamma > 0 is always a vertex of the upper convex hull of the
points (Pen, G), taken from the point of largest G (gamma -> 0) to the point of smallest Pen
(gamma -> infinity). So an intermediate-size optimum is possible for SOME gamma exactly when an
intermediate-size pattern is a vertex of that hull, and this script counts, per sum-mode cell:

  - whether such a vertex exists, the gamma interval on which it is the maximiser, and the
    largest F it reaches there (an interior optimum bought where F <= 0 is the penalty having
    consumed the reward);
  - for each gamma of the study's grid, whether the maximiser has an intermediate edge count.

"Intermediate" is counted under three definitions:
  - pool: an edge count strictly between the smallest and the largest edge count of the pool, the
    manuscript's definition since 2026-11-20 (ticket 4);
  - edges: 1 < |E| < K, the reading of run_degeneracy.shape_of and of degeneracy_gamma_*.csv;
  - bound: 1 < |E| and neither |E| = K nor |V| = K, the manuscript's earlier definition.
The first two differ only in a cell whose pool has no pattern with K edges (two cells at K = 4,
where a 4-edge pattern needs a cycle the molecules do not have).

The cells are those of results/degeneracy_frontier_*.csv. Before writing anything the script
re-derives, for every cell and every gamma of the grid, the edge count of the maximiser and
refuses unless it equals peak_edges in results/degeneracy_gamma_*.csv.

Pools nest (a pattern admitted under a tight bound or a high threshold is admitted under a
looser one, tests/test_pool_nesting.py), so each collection is mined once at its loosest cell
and the cells are filters over that pool, as scripts/run_pareto_size.py does.

Counts only, no timings.

    python scripts/hull_interior_check.py
    python scripts/hull_interior_check.py --out results-invariant/hull_interior.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_degeneracy as rd                                       # noqa: E402
from wfsmbench.baselines import mine_exhaustive                   # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params         # noqa: E402
from wfsmbench.io import load_dataset                             # noqa: E402
from wfsmbench.measure.fitness import fitness, penalty            # noqa: E402
from wfsmbench.preprocess import build_statistics                 # noqa: E402

RESULTS = ROOT / "results"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def upper_hull(points: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Upper convex hull, left to right, of (Pen, G, index) points; best G kept per Pen."""
    best: dict[float, tuple[float, float, int]] = {}
    for pen, g, i in points:
        if pen not in best or g > best[pen][1]:
            best[pen] = (pen, g, i)
    hull: list[tuple[float, float, int]] = []
    for p in sorted(best.values()):
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            # drop the middle point unless it lies strictly above the chord
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results-invariant" / "hull_interior.csv")
    args = ap.parse_args(argv)

    cells = sorted({(r["dataset"], int(r["graph_limit"]), int(r["max_size"]), int(r["min_support"]))
                    for p in sorted(RESULTS.glob("degeneracy_frontier_*.csv")) for r in rows(p)})
    tracked = defaultdict(dict)
    for p in sorted(RESULTS.glob("degeneracy_gamma_*.csv")):
        for r in rows(p):
            tracked[(r["dataset"], int(r["graph_limit"]), int(r["max_size"]),
                     int(r["min_support"]))][float(r["gamma"])] = int(r["peak_edges"])
    if not cells:
        print("[fail] no frontier cells in results/")
        return 1
    print(f"{len(cells)} sum-mode cells; gamma grid {rd.GAMMAS}", flush=True)

    by_collection = defaultdict(list)
    for cell in cells:
        by_collection[(cell[0], cell[1])].append(cell)

    out, compared, mismatches = [], 0, []
    for (dataset, limit), group in sorted(by_collection.items()):
        graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
        subset = graphs[:limit]
        stats = build_statistics(subset)
        loose = Params(max_nodes=max(c[2] for c in group), max_edges=max(c[2] for c in group),
                       min_support=min(c[3] for c in group), size_reference=STUDY_SIZE_REFERENCE)
        pool = mine_exhaustive(subset, stats, loose).pool
        print(f"{dataset} n={limit}: loose pool {len(pool)}", flush=True)
        for _d, _l, size, support in group:
            base = Params(max_nodes=size, max_edges=size, min_support=support,
                          size_reference=STUDY_SIZE_REFERENCE)
            cell_pool = [c for c in pool if c.particle.pattern.num_edges() <= size
                         and c.particle.pattern.num_nodes() <= size
                         and c.evaluation.support >= support]
            free = Params(max_nodes=size, max_edges=size, min_support=support,
                          size_reference=STUDY_SIZE_REFERENCE, gamma=0.0)
            pts, edges, nodes = [], [], []
            for i, c in enumerate(cell_pool):
                v, k = c.particle.pattern.num_nodes(), c.particle.pattern.num_edges()
                g = fitness(c.evaluation, v, k, stats, free)
                pts.append((penalty(v, k, base), g, i))
                edges.append(k)
                nodes.append(v)
            lo, hi = min(edges), max(edges)

            def inner_pool(i):
                return lo < edges[i] < hi

            def inner_edges(i):
                return lo < edges[i] < size

            def inner_bound(i):
                return lo < edges[i] and edges[i] != size and nodes[i] != size

            grid_inside = grid_inside_bound = grid_inside_pool = 0
            grid_peaks = []
            for gamma in rd.GAMMAS:
                tuned = Params(max_nodes=size, max_edges=size, min_support=support,
                               size_reference=STUDY_SIZE_REFERENCE, gamma=gamma)
                scores = [fitness(c.evaluation, c.particle.pattern.num_nodes(),
                                  c.particle.pattern.num_edges(), stats, tuned) for c in cell_pool]
                top = max(scores)
                winners = [i for i, s in enumerate(scores) if s == top]
                peak_edges = {edges[i] for i in winners}
                # A tie between edge counts is reported as the smallest; the comparison with
                # the tracked sweep accepts any edge count that reaches the maximum. A tie
                # counts as intermediate only if every tied maximiser is.
                peak = min(peak_edges)
                grid_peaks.append(f"{peak}e{min(nodes[i] for i in winners if edges[i] == peak)}v")
                grid_inside += all(inner_edges(i) for i in winners)
                grid_inside_bound += all(inner_bound(i) for i in winners)
                grid_inside_pool += all(inner_pool(i) for i in winners)
                want = tracked.get((dataset, limit, size, support), {}).get(gamma)
                if want is not None:
                    compared += 1
                    if want not in peak_edges:
                        mismatches.append((dataset, limit, size, support, gamma, sorted(peak_edges), want))

            hull = upper_hull(pts)
            g_best = max(range(len(hull)), key=lambda j: hull[j][1])
            reachable = hull[:g_best + 1]          # from smallest Pen up to the largest G
            vertices = []
            for j, (pen, g, i) in enumerate(reachable):
                # gamma interval on which this vertex is the maximiser: slopes to its neighbours
                lo_g = 0.0 if j == len(reachable) - 1 else \
                    (reachable[j + 1][1] - g) / (reachable[j + 1][0] - pen)
                hi_g = float("inf") if j == 0 else (g - reachable[j - 1][1]) / (pen - reachable[j - 1][0])
                vertices.append((i, lo_g, hi_g, g - lo_g * pen))
            out_row = {}
            for label, test in (("pool", inner_pool), ("edges", inner_edges), ("bound", inner_bound)):
                inner = [x for x in vertices if test(x[0])]
                out_row[f"intermediate_vertices_{label}"] = len(inner)
                out_row[f"intermediate_gamma_{label}"] = (
                    f"{min(x[1] for x in inner):.6f}-{max(x[2] for x in inner):.6f}" if inner else "")
                out_row[f"intermediate_best_F_{label}"] = (
                    f"{max(x[3] for x in inner):.6f}" if inner else "")
            out.append({"dataset": dataset, "graphs": limit, "size_bound": size,
                        "min_support": support, "pool": len(cell_pool),
                        "hull_vertices": len(vertices),
                        "hull_vertex_shapes": "|".join(f"{edges[x[0]]}e{nodes[x[0]]}v" for x in vertices),
                        **out_row,
                        "grid_gammas": len(rd.GAMMAS),
                        "grid_peaks": "|".join(grid_peaks),
                        "grid_intermediate_pool": grid_inside_pool,
                        "grid_intermediate_edges": grid_inside,
                        "grid_intermediate_bound": grid_inside_bound,
                        "pool_edges_max": hi,
                        "grid_peak_at_default_gamma_intermediate_pool": int(
                            lo < int(grid_peaks[0].split("e")[0]) < hi)})
            r = out[-1]
            print(f"  K={size} s={support}: pool {len(cell_pool)}, hull {r['hull_vertex_shapes']}; "
                  f"intermediate vertices edges/bound {r['intermediate_vertices_edges']}/"
                  f"{r['intermediate_vertices_bound']} (best F {r['intermediate_best_F_edges'] or '-'}/"
                  f"{r['intermediate_best_F_bound'] or '-'}); grid {r['grid_peaks']}", flush=True)

    print(f"[reproduced] {compared} (cell, gamma) peaks compared with degeneracy_gamma_*.csv, "
          f"{len(mismatches)} mismatch(es)")
    if compared == 0 or mismatches:
        for m in mismatches[:10]:
            print(f"  [mismatch] {m}")
        print("[refused] the re-derived peaks do not match the tracked gamma sweep; nothing written")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    for label in ("pool", "edges", "bound"):
        with_vertex = sum(1 for r in out if r[f"intermediate_vertices_{label}"])
        with_grid = sum(1 for r in out if r[f"grid_intermediate_{label}"])
        positive = sum(1 for r in out if r[f"intermediate_best_F_{label}"]
                       and float(r[f"intermediate_best_F_{label}"]) > 0)
        print(f"\n[{label}] {with_vertex} of {len(out)} cells have an intermediate vertex on the upper "
              f"hull ({positive} with F > 0 there); {with_grid} of {len(out)} have an intermediate "
              f"maximiser at some gamma of the grid")
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
