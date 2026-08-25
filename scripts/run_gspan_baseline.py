r"""gSpan against this project's own enumerator, head to head in one process.

The published algorithm and the enumerator written here answer the same question -- every
frequent connected subgraph up to a size bound -- so their cost is comparable and a reviewer
asking for a published baseline is asking for exactly this number.

**Both sides are timed inside one process, on one cell.** Taking gSpan's seconds from one run
and the enumerator's from another would be arithmetic across runs: different caches, different
heap, different warm-up. A speed-up assembled that way is not a measurement, and this project
has a rule against it because it produced a wrong conclusion once already. Where the slow side
is skipped its column is left **empty** rather than filled from an earlier log, so nothing can
quietly compare across runs.

This is a timing measurement, so it belongs on the measurement machine and nowhere else.

The optimum is also recomputed from gSpan's output and written beside the enumerator's, which
is a second, free correctness check: the two must name a pattern of the same size.

    python scripts/run_gspan_baseline.py --dataset Cuneiform --sizes 4 5 --suffix _Cuneiform
    python scripts/run_gspan_baseline.py --dataset Cuneiform --sizes 6 --skip-exhaustive ^
        --suffix _Cuneiform_s6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.baselines import mine_exhaustive                      # noqa: E402
from wfsmbench.baselines.gspan import code_to_graph, gspan           # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params            # noqa: E402
from wfsmbench.core.signature import canonical_signature             # noqa: E402
from wfsmbench.io import load_dataset                                # noqa: E402
from wfsmbench.match import match_database                           # noqa: E402
from wfsmbench.measure.evaluation import Evaluation                   # noqa: E402
from wfsmbench.measure.fitness import fitness                         # noqa: E402
from wfsmbench.preprocess import build_statistics                     # noqa: E402

RESULTS = ROOT / "results"
# Counts and set comparisons do not depend on the machine, so they are kept apart
# from the timed artefacts and carry no timing column at all.
INVARIANT = ROOT / "results-invariant"
STUDY_GRAPH_LIMIT = 60

HEADER = ("dataset,graph_limit,size_bound,min_support,"
          "gspan_patterns,gspan_seconds,"
          "exhaustive_patterns,exhaustive_seconds,speedup,"
          "sets_agree,supports_disagreeing,"
          "gspan_optimum_edges,gspan_optimum_fitness")


def score(patterns, hosts, stats, params) -> tuple[int, float]:
    """Best fitness over a set of patterns, and the edge count of the winner."""
    best_edges, best_value = 0, None
    for pattern in patterns:
        evaluation = Evaluation.from_embeddings(match_database(pattern, hosts))
        if evaluation.support < params.min_support:
            continue
        value = fitness(evaluation, pattern.num_nodes(), pattern.num_edges(), stats, params)
        if best_value is None or value > best_value:
            best_edges, best_value = pattern.num_edges(), value
    return best_edges, (best_value if best_value is not None else 0.0)


def agreement(subset, stats, bound: int, min_support: int, skip_exhaustive: bool) -> dict:
    """Mine both ways and compare the pattern sets. Counts here; timings are the caller's.

    Splitting this out is what lets the same implementation serve two artefacts: the counts
    do not depend on the machine and belong in results-invariant/, while any timing must be
    measured on the machine the paper names. One computation, two outputs -- writing a second
    comparison script would give the project two definitions of "agree".
    """
    params = Params(min_support=min_support, max_nodes=bound, max_edges=bound,
                    size_reference=STUDY_SIZE_REFERENCE)

    started = time.perf_counter()
    mined = gspan(subset, min_support, bound, bound)
    gspan_seconds = time.perf_counter() - started
    patterns = [code_to_graph(code) for code, _ in mined]
    supports = {canonical_signature(g): s for g, (_c, s) in zip(patterns, mined)}

    found = {"gspan_patterns": len(mined), "gspan_seconds": gspan_seconds,
             "patterns": patterns, "params": params}
    if skip_exhaustive:
        found.update(exhaustive_patterns="", exhaustive_seconds=None, agree="",
                     disagreeing="", missing="", extra="")
        return found

    started = time.perf_counter()
    pool = mine_exhaustive(subset, stats, params).pool
    found["exhaustive_seconds"] = time.perf_counter() - started
    reference = {canonical_signature(c.particle.pattern): c.evaluation.support for c in pool}
    shared = set(reference) & set(supports)
    found.update(
        exhaustive_patterns=str(len(reference)),
        agree=str(int(set(reference) == set(supports))),
        # Reported with its denominator: "0 disagreeing" over an empty intersection would
        # be no check at all.
        disagreeing=f"{sum(1 for k in shared if reference[k] != supports[k])}/{len(shared)}",
        missing=str(len(set(reference) - set(supports))),
        extra=str(len(set(supports) - set(reference))))
    return found


def one_cell(subset, stats, dataset: str, bound: int, min_support: int,
             skip_exhaustive: bool) -> str:
    found = agreement(subset, stats, bound, min_support, skip_exhaustive)
    gspan_seconds = found["gspan_seconds"]
    if found["exhaustive_seconds"] is None:
        exhaustive_seconds = speedup = ""
    else:
        exhaustive_seconds = f"{found['exhaustive_seconds']:.3f}"
        speedup = (f"{found['exhaustive_seconds'] / gspan_seconds:.2f}"
                   if gspan_seconds else "")

    best_edges, best_value = score(found["patterns"], subset, stats, found["params"])
    return (f"{dataset},{len(subset)},{bound},{min_support},"
            f"{found['gspan_patterns']},{gspan_seconds:.3f},"
            f"{found['exhaustive_patterns']},{exhaustive_seconds},{speedup},"
            f"{found['agree']},{found['disagreeing']},{best_edges},{best_value:.6f}")


INVARIANT_HEADER = ("dataset,graph_limit,size_bound,min_support,"
                    "exhaustive_patterns,gspan_patterns,sets_agree,"
                    "supports_disagreeing_over_shared,missing_from_gspan,extra_in_gspan")


def counts_only(subset, stats, args) -> int:
    """The comparison a reader can reproduce on any machine: do the two agree, and over how
    many patterns. The enumerator is never skipped here -- with nothing to compare against,
    the file would record a count and call it an agreement."""
    rows = [INVARIANT_HEADER]
    for bound in args.sizes:
        print(f"  {args.dataset} bound={bound} support={args.support} ...", flush=True)
        found = agreement(subset, stats, bound, args.support, skip_exhaustive=False)
        rows.append(f"{args.dataset},{len(subset)},{bound},{args.support},"
                    f"{found['exhaustive_patterns']},{found['gspan_patterns']},"
                    f"{found['agree']},{found['disagreeing']},"
                    f"{found['missing']},{found['extra']}")
        print(f"    enumerator {found['exhaustive_patterns']:>6} | gSpan "
              f"{found['gspan_patterns']:>6} | sets agree {found['agree']} | "
              f"supports disagreeing {found['disagreeing']}", flush=True)

    INVARIANT.mkdir(exist_ok=True)
    path = INVARIANT / f"gspan_agreement{args.suffix}.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[ok] {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Cuneiform", help="dataset JSON stem in data/")
    ap.add_argument("--graph-limit", type=int, default=STUDY_GRAPH_LIMIT)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--support", type=int, default=2,
                    help="absolute minimum support; 2 of 60 graphs is the study's loosest")
    ap.add_argument("--skip-exhaustive", action="store_true",
                    help="run gSpan only and leave the comparison columns EMPTY; use where "
                         "the enumerator would take hours, never to fill them from a log")
    ap.add_argument("--counts-only", action="store_true",
                    help="write the pattern-set agreement to results-invariant/ with NO "
                         "timing column; those counts do not depend on the machine")
    ap.add_argument("--suffix", required=True,
                    help="required: an unsuffixed name lets one collection overwrite another")
    args = ap.parse_args()

    graphs, _ = load_dataset(ROOT / "data" / f"{args.dataset}.json")
    subset = graphs[:args.graph_limit]
    stats = build_statistics(subset)

    if args.counts_only:
        return counts_only(subset, stats, args)

    rows = [HEADER]
    for bound in args.sizes:
        print(f"  {args.dataset} bound={bound} support={args.support} ...", flush=True)
        row = one_cell(subset, stats, args.dataset, bound, args.support,
                       args.skip_exhaustive)
        rows.append(row)
        parts = row.split(",")
        print(f"    gSpan {parts[4]:>6} patterns in {float(parts[5]):8.1f}s"
              + (f" | enumerator {parts[6]:>6} in {float(parts[7]):8.1f}s"
                 f" = {parts[8]}x | sets agree {parts[9]}, supports disagreeing {parts[10]}"
                 if parts[7] else " | enumerator skipped")
              + f" | optimum {parts[11]} edges", flush=True)

    path = RESULTS / f"gspan_baseline{args.suffix}.csv"
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
        if len(existing) > 1:
            before, after = tuple(existing[1].split(",")[:2]), tuple(rows[1].split(",")[:2])
            if before != after:
                raise SystemExit(
                    f"[refused] {path} holds {before[0]} n={before[1]}; this run writes "
                    f"{after[0]} n={after[1]}. Delete it if superseded, or pass a different "
                    f"--suffix if both are wanted.")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n[ok] {path.relative_to(ROOT)}  ({len(rows) - 1} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
