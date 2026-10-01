"""Measure the basin structure of the optimum, an instance-level property.

Two controlled arms have now shown that aggregate landscape statistics do not decide
which method wins. Holding every one of them equal across three collections left the
ranking free to reverse significantly; forcing the plateau structure of one dataset
through a twenty-one-fold change moved no method by more than a fifth of a critical
difference. Whatever selects the winner is therefore not a summary of the fitness
distribution, and the natural next place to look is where the optimum *sits* rather
than how many distinct values surround it.

This measures that. From every pattern in the enumerated pool, follow steepest ascent
along single-edit moves that stay inside the pool, and record where it stops. What
comes out is not a distribution summary but a statement about reachability:

    basin_optimum_pct   share of the pool whose ascent ends on an optimal pattern.
                        The direct answer to "how findable is the best answer by local
                        improvement" -- small means a method must do something other
                        than improve locally, which is what separates the algorithms.
    attractors          distinct patterns the ascents terminate on.
    largest_basin_pct   share of the pool draining into the single biggest attractor,
                        and optimum_is_largest says whether that attractor is optimal.
                        These two apart is the interesting case: most of the space
                        pulling somewhere other than the answer.
    mean_path           average ascent steps taken, a scale for how far apart things are.

Ties are handled explicitly because the choice biases the result and hiding it would be
the kind of decision that looks like preprocessing. Strict ascent stops when no
neighbour is strictly fitter, so on a plateau it halts immediately; plateau-walking
also accepts equal neighbours, so it drifts across a plateau until something fitter or
nothing new appears. The two differ only where ties exist, and both are reported --
Cuneiform is full of ties and the molecular collections have almost none, so a single
policy would describe one family and mislead about the other.

Deterministic throughout: neighbours are ordered by canonical signature before any
choice, so no run-to-run variation and no dependence on enumeration order.

    python scripts/run_basins.py --dataset Cuneiform --limit 60 --sizes 5 \
        --supports 2 --modes sum --suffix _cuneiform
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_weight_mode as rwm                                  # noqa: E402
from wfsmbench.core.signature import canonical_signature      # noqa: E402
from wfsmbench.io import load_dataset                          # noqa: E402
from wfsmbench.preprocess import build_statistics              # noqa: E402
from wfsmbench.pso.optimizer import _neighbors                 # noqa: E402

RESULTS = ROOT / "results"
TOLERANCE = 1e-12

# graph_offset is part of the cell identity, not a note. A slice and the base run
# share their dataset and mode, so anything aggregating across files keyed on those
# two silently merges them -- which happened: a development-box slice of QM9 sat
# beside the experiment-machine run in the same table, both claiming to be QM9 at
# two hundred graphs, with different pools and different basins.
HEADER = ("dataset,graph_limit,graph_offset,max_size,min_support,weight_mode,fitness_quantum,"
          "pool_size,optimum_multiplicity,policy,attractors,basin_optimum_pct,"
          "largest_basin_pct,optimum_is_largest,mean_path,max_path,"
          "unreached_from_pct")


def neighbour_signatures(candidate, frequent, params, scored) -> list[str]:
    """In-pool single-edit neighbours, ordered so every later choice is deterministic."""
    seen = {canonical_signature(n.pattern) for n in _neighbors(candidate.particle, frequent,
                                                               params)}
    return sorted(sig for sig in seen if sig in scored)


def ascend(start: str, table: dict[str, float], graph: dict[str, list[str]],
           walk_plateaus: bool) -> tuple[str, int]:
    """Follow steepest ascent from `start`; return where it stops and how far it went.

    Under plateau-walking a visited set bounds the drift, so a flat region cannot cycle.
    Ties among equally good moves are broken by signature order, which is why the
    neighbour lists are sorted: without that the answer would depend on the order the
    enumerator happened to produce.
    """
    current, steps, visited = start, 0, {start}
    while True:
        best, best_value = None, table[current]
        for sig in graph[current]:
            value = table[sig]
            if value > best_value + TOLERANCE:
                best, best_value = sig, value
        if best is None and walk_plateaus:
            for sig in graph[current]:
                if abs(table[sig] - table[current]) <= TOLERANCE and sig not in visited:
                    best = sig
                    break
        if best is None:
            return current, steps
        current, steps = best, steps + 1
        visited.add(current)


def measure(pool, stats, params, policy: str) -> dict:
    scored = {}
    for candidate in pool:
        pattern = candidate.particle.pattern
        scored[candidate.signature] = rwm.score_fitness(
            candidate.evaluation, pattern.num_nodes(), pattern.num_edges(), stats, params)

    frequent = set(stats.frequent_edge_templates(params.min_support))
    graph = {c.signature: neighbour_signatures(c, frequent, params, scored) for c in pool}

    best_value = max(scored.values())
    optimum = {sig for sig, value in scored.items() if value >= best_value - TOLERANCE}

    ends: dict[str, int] = {}
    reached_optimum = paths = longest = 0
    for signature in scored:
        stop, steps = ascend(signature, scored, graph, policy == "plateau")
        ends[stop] = ends.get(stop, 0) + 1
        paths += steps
        longest = max(longest, steps)
        if stop in optimum:
            reached_optimum += 1

    size = len(scored)
    biggest = max(ends.items(), key=lambda item: item[1])
    # A pattern nothing ascends into is one no local improvement can arrive at from
    # anywhere else; a space made mostly of these is one where a method's starting points
    # decide its answer.
    unreached = sum(1 for sig in scored if sig not in ends)
    return {
        "policy": policy,
        "attractors": len(ends),
        "basin_optimum_pct": 100.0 * reached_optimum / size,
        "largest_basin_pct": 100.0 * biggest[1] / size,
        "optimum_is_largest": int(biggest[0] in optimum),
        "mean_path": paths / size,
        "max_path": longest,
        "unreached_from_pct": 100.0 * unreached / size,
        "optimum_multiplicity": len(optimum),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many graphs before taking --limit. QM9 and alchemy are "
                         "sorted by molecule size and hold six figures of graphs, so a later "
                         "slice is a different instance -- different molecules, pool, optimum "
                         "and basin -- drawn from the same population")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sizes", default="5")
    ap.add_argument("--supports", default="2")
    ap.add_argument("--modes", default="sum")
    ap.add_argument("--rho", type=float, default=None,
                    help="discount to search under; without it the calibrated value for "
                         "this collection is read from the artefacts, and a collection "
                         "with none refuses to run under discounting rather than "
                         "silently taking the default")
    ap.add_argument("--quantum", type=float, default=0.0)
    ap.add_argument("--suffix", required=True,
                    help="required, so two datasets can never write the same file")
    args = ap.parse_args(argv)

    graphs, _ = load_dataset(ROOT / "data" / f"{args.dataset}.json")
    start = args.offset
    subset = graphs[start:] if args.limit is None else graphs[start:start + args.limit]
    stats = build_statistics(subset)
    modes = args.modes.split(",")
    rho = rwm.resolve_rho(args.dataset, modes, args.rho)
    print(f"{args.dataset} n={len(subset)}"
          + (f", rho = {rho}" if "discounted" in modes else ""), flush=True)

    rows = [HEADER]
    for size in [int(s) for s in args.sizes.split(",")]:
        for support in [int(s) for s in args.supports.split(",")]:
            for mode in args.modes.split(","):
                params = rwm.params_for(size, rwm.BUDGETS[0], mode, support,
                                        False, rho, args.quantum)
                pool = rwm.enumerate_pool(subset, stats, params)
                if not pool:
                    print(f"  size {size} supp {support} [{mode}]: pool empty", flush=True)
                    continue
                for policy in ("strict", "plateau"):
                    info = measure(pool, stats, params, policy)
                    rows.append(
                        f"{args.dataset},{len(subset)},{args.offset},{size},{support},{mode},"
                        f"{args.quantum:g},{len(pool)},{info['optimum_multiplicity']},"
                        f"{policy},{info['attractors']},{info['basin_optimum_pct']:.3f},"
                        f"{info['largest_basin_pct']:.3f},{info['optimum_is_largest']},"
                        f"{info['mean_path']:.3f},{info['max_path']},"
                        f"{info['unreached_from_pct']:.3f}")
                    print(f"  size {size} supp {support} [{mode}] {policy:<8}: "
                          f"pool {len(pool)}, {info['attractors']} attractors, "
                          f"optimum basin {info['basin_optimum_pct']:.1f}%, "
                          f"largest {info['largest_basin_pct']:.1f}% "
                          f"({'is' if info['optimum_is_largest'] else 'is NOT'} the optimum)",
                          flush=True)

    out = RESULTS / f"basins{args.suffix}.csv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[done] {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake      # Windows: no sleep while the stage runs
    keep_awake()
    raise SystemExit(main(sys.argv[1:]))
