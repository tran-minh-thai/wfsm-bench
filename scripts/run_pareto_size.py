r"""Where a Pareto front puts the answer on the size axis.

The obvious rejoinder to a degenerate scalarised objective is to stop scalarising: keep
(WSUP, ATW, -size) as a vector and report the non-dominated set, as multi-objective subgraph
miners have done since 2014. That removes the need to weight the terms. It does not remove
the need to *answer*, because a front is a set and a practitioner wants a pattern.

This measures what happens at that last step. On the exhaustively enumerated frequent pool it
computes the true front, then applies selectors that a practitioner would actually reach for,
and records the pattern size each one lands on relative to the sizes present on the front.

Three selectors, chosen so that two of them owe nothing to the objective under study:

    ideal     min-max normalise the front, take the point closest to (1, 1, 1).
              Parameter-free. This is the usual compromise-programming pick.
    maxmin    maximise the smallest normalised objective (Chebyshev). Parameter-free,
              and unlike `ideal` it is insensitive to how many objectives agree.
    weighted  the study's objective F itself, evaluated on each front member against the
              study's fixed anchors -- not against the front's own extremes. Normalising
              against the front would re-anchor on a run-dependent quantity, which is the
              failure this project exists to document; doing it here would measure
              something else and call it F.

`weighted` is reported for contrast and is circular by construction -- selecting a front by
the very scalarisation whose optimum is under study can only return that optimum. It is here
because that circularity is the point: a reader who proposes "go multi-objective" has to say
which point, and the most familiar answer walks straight back into the thing it escaped.

Reported per configuration: the pool and front sizes, the distinct pattern sizes present on
the front, and for each selector the size it picked and whether that size is the front's
largest, its smallest, or neither. Nothing here is a timing measurement, so it is a property
of the data and the code rather than of the machine that ran it, and it needs no designated
measurement machine.

**One enumeration, twenty-one cells.** Mining each cell separately re-walks overlapping
search spaces, and the largest cell alone is 42,523 patterns. The pools nest: `is_valid`
rejects on the size bounds so a pattern admitted under a tight bound is admitted under a
loose one; `frequent_edge_templates` returns more templates as the threshold falls and only
frequent patterns are expanded, so a low threshold explores a superset; and `fitness` anchors
on `Params.size_reference`, never on `max_edges`, so a pattern's F does not depend on the
bound it was found under. This script therefore mines once at the loosest setting and derives
every cell by filtering. `tests/test_pool_nesting.py` holds that to a direct mining -- same
patterns, same support, same fitness -- rather than leaving it as three readings of the
source.

    python scripts/run_pareto_size.py --dataset cuneiform --suffix _cuneiform
    python scripts/run_pareto_size.py --dataset cuneiform --suffix _cuneiform --sizes 4 5 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.baselines import mine_exhaustive                  # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params        # noqa: E402
from wfsmbench.io import load_dataset                            # noqa: E402
from wfsmbench.measure.pareto import bounds_of, normalize, objectives, pareto_front  # noqa: E402
from wfsmbench.preprocess import build_statistics                # noqa: E402

RESULTS = ROOT / "results"
SUPPORT_FRACTIONS = (1 / 30, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70)

# The 21 cells of the degeneracy result were run on 60 graphs, not the whole collection:
# every min_support in results/weight_mode_landscape.csv is a fraction of 60, and its
# graph_limit column says 60. Running the whole 267 answers a different question under the
# same cell names, and costs about four times as much.
STUDY_GRAPH_LIMIT = 60

# Two size columns on purpose. The objective vector negates |V| + |E|, but the size-degeneracy
# analysis counts in edges, and the two disagree: a four-edge pattern is size 9 as a tree and size 8 as a
# cycle, so a pick at the edge bound can look interior on |V| + |E|. Both are recorded, and
# the edge columns are the ones comparable with that analysis.
HEADER = ("dataset,graph_limit,size_bound,support_fraction,min_support,"
          "pool_size,front_size,front_sizes,front_min_size,front_max_size,"
          "front_edge_range,front_min_edges,front_max_edges,"
          "selector,picked_size,picked_edges,"
          "at_front_max_size,at_front_min_size,interior_size,"
          "at_edge_bound,at_front_max_edges,at_front_min_edges,interior_edges")


def size_of(point: tuple[float, ...]) -> int:
    """Recover |V| + |E| from the negated size objective."""
    return int(round(-point[2]))


def pick_ideal(scaled: list[tuple[float, ...]]) -> int:
    """Index of the point closest to the ideal corner (1, 1, 1)."""
    return min(range(len(scaled)),
               key=lambda i: sum((1.0 - v) ** 2 for v in scaled[i]))


def pick_maxmin(scaled: list[tuple[float, ...]]) -> int:
    """Index maximising the smallest normalised objective."""
    return max(range(len(scaled)), key=lambda i: min(scaled[i]))


def pick_weighted(members: list, params: Params) -> int:
    """Index of the front member with the largest study fitness F, fixed anchors."""
    return max(range(len(members)), key=lambda i: members[i].fitness)


def rows_for(cell_pool, params: Params, dataset: str, graph_limit: int, size_bound: int,
             fraction: float, min_support: int) -> list[str]:
    pool = cell_pool
    if not pool:
        return []
    points = [objectives(c.evaluation, c.particle.pattern) for c in pool]
    front_points = set(pareto_front(points))
    if not front_points:
        return []

    # Keep the candidates, not only their objective vectors: F has to be read off the
    # candidate against the study anchors, and the pattern's edge count is what the size-degeneracy
    # analysis counts in. One representative per distinct objective vector, chosen deterministically.
    members: list = []
    seen: set[tuple[float, ...]] = set()
    for candidate, point in sorted(zip(pool, points), key=lambda cp: cp[1], reverse=True):
        if point in front_points and point not in seen:
            seen.add(point)
            members.append(candidate)
    front = [objectives(c.evaluation, c.particle.pattern) for c in members]

    sizes = sorted({size_of(p) for p in front})
    edge_counts = sorted({c.particle.pattern.num_edges() for c in members})
    scaled = normalize(front, bounds_of(front))
    picks = {"ideal": pick_ideal(scaled),
             "maxmin": pick_maxmin(scaled),
             "weighted": pick_weighted(members, params)}

    out: list[str] = []
    for name, index in picks.items():
        picked = size_of(front[index])
        edges = members[index].particle.pattern.num_edges()
        out.append(
            f"{dataset},{graph_limit},{size_bound},{fraction:.6f},{min_support},"
            f"{len(pool)},{len(front)},{'|'.join(str(s) for s in sizes)},"
            f"{sizes[0]},{sizes[-1]},"
            f"{'|'.join(str(e) for e in edge_counts)},{edge_counts[0]},{edge_counts[-1]},"
            f"{name},{picked},{edges},"
            f"{int(picked == sizes[-1])},{int(picked == sizes[0])},"
            f"{int(sizes[0] < picked < sizes[-1])},"
            f"{int(edges == size_bound)},{int(edges == edge_counts[-1])},"
            f"{int(edges == edge_counts[0])},"
            f"{int(edge_counts[0] < edges < edge_counts[-1])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cuneiform",
                    help="dataset JSON stem in data/")
    ap.add_argument("--graph-limit", type=int, default=STUDY_GRAPH_LIMIT,
                    help=f"graphs to use; default {STUDY_GRAPH_LIMIT}, which is what the "
                         "degeneracy result's 21 cells were run on. 0 uses the whole "
                         "collection and answers a different question.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--supports", type=float, nargs="+", default=None,
                    help="support fractions; default is the study's seven")
    ap.add_argument("--suffix", required=True,
                    help="required: an unsuffixed name lets one collection overwrite another")
    args = ap.parse_args()

    graphs, _ = load_dataset(ROOT / "data" / f"{args.dataset}.json")
    limit = args.graph_limit or len(graphs)
    subset = graphs[:limit]
    stats = build_statistics(subset)
    fractions = args.supports if args.supports else list(SUPPORT_FRACTIONS)

    loose_bound = max(args.sizes)
    loose_support = max(1, int(round(min(fractions) * len(subset))))
    # max_nodes AND max_edges both equal the bound, because that is what
    # run_weight_mode.py does and therefore what the 21 cells of the degeneracy result
    # mean. Allowing one extra node quadrupled the pool at bound 4 -- 2959 against 712 --
    # and would have measured a different set of cells while reporting the same names.
    loose_params = Params(min_support=loose_support, max_nodes=loose_bound,
                          max_edges=loose_bound, size_reference=STUDY_SIZE_REFERENCE)
    print(f"[mine] one enumeration covers every cell: bound={loose_bound} "
          f"min_support={loose_support} over {len(subset)} graphs", flush=True)
    started = time.perf_counter()
    loose_pool = mine_exhaustive(subset, stats, loose_params).pool
    print(f"[mine] {len(loose_pool)} patterns in {time.perf_counter() - started:.1f}s; "
          f"the {len(args.sizes) * len(fractions)} cells are filters over it\n", flush=True)

    rows = [HEADER]
    for size_bound in args.sizes:
        for fraction in fractions:
            min_support = max(1, int(round(fraction * len(subset))))
            cell = [c for c in loose_pool
                    if c.particle.pattern.num_edges() <= size_bound
                    and c.particle.pattern.num_nodes() <= size_bound
                    and c.evaluation.support >= min_support]
            params = Params(min_support=min_support, max_nodes=size_bound,
                            max_edges=size_bound, size_reference=STUDY_SIZE_REFERENCE)
            produced = rows_for(cell, params, args.dataset, limit, size_bound,
                                fraction, min_support)
            if not produced:
                print(f"  [empty] bound={size_bound} support={fraction:.3f}: "
                      f"no frequent pool, no front -- cell reported as absent, not as zero",
                      flush=True)
                continue
            rows.extend(produced)
            head = produced[0].split(",")
            print(f"  bound={size_bound} support={fraction:.3f}  pool={head[5]:>6} "
                  f"front={head[6]:>3}  front edges={head[10]:<12} "
                  + "  ".join(f"{r.split(',')[13]}->{r.split(',')[15]}e" for r in produced),
                  flush=True)

    if len(rows) < 2:
        print("[fail] no configuration produced a front; nothing written", file=sys.stderr)
        return 1

    path = RESULTS / f"pareto_size{args.suffix}.csv"
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
        if len(existing) > 1:
            before, after = tuple(existing[1].split(",")[:2]), tuple(rows[1].split(",")[:2])
            if before != after:
                raise SystemExit(
                    f"[refused] {path} holds {before[0]} n={before[1]}; this run writes "
                    f"{after[0]} n={after[1]}.\n"
                    f"          Two columns disagree, so this is a replacement and not a "
                    f"re-run, and overwriting would destroy the first while leaving its "
                    f"name in place.\n"
                    f"          If the existing file describes a configuration that has "
                    f"since been corrected, delete it and run again -- that is a deliberate "
                    f"act, which is the point.\n"
                    f"          If both configurations are wanted, pass a different "
                    f"--suffix so each keeps its own file.")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n[ok] {path.relative_to(ROOT)}  ({len(rows) - 1} rows)")
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake      # Windows: no sleep while the stage runs
    keep_awake()
    raise SystemExit(main())
