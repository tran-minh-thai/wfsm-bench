"""PROBE, not an experiment. Can an upper bound prune the Pareto-front search at all?

ANSWER, measured 2026-08-23: no, not usefully. Kept so the finding is reproducible and so
nobody spends a week implementing the miner this was meant to justify.

The bound. For a pattern g with support s, largest embedding weight maxW, k edges and v
nodes, and a size bound K, every extension h of g satisfies:

    s(h)     <= s(g)                        support is anti-monotone
    a(h)     <= maxW(g) + (K - k)*w_max     an h-embedding contains a g-embedding plus at
                                            most K-k further edges, each at most w_max, and
                                            a mean is at most a maximum
    WSUP(h)  <= s(g) * [maxW(g) + (K-k)*w_max]
    -size(h) <= -size(g)                    size only grows

so U(g) = (that WSUP bound, that ATW bound, -size(g)) dominates the objective vector of
every extension componentwise. If some front member f dominates U(g), then f dominates every
extension of g, and the whole subtree can be cut.

This measures the OPTIMISTIC ceiling: it prunes against the TRUE front, computed in advance
by full enumeration. A real miner has only a partial front and can never do better. If the
ceiling is low, the idea is dead and no implementation is worth writing.

Reports counts only -- no timings -- so the numbers do not depend on the machine.

WHAT IT FOUND. Pruning is **zero at k=1 and k=2 in every collection tried**, and only fires
within one level of the size bound:

    collection           w_max/mean   prunable at k=3 (bound 4, not leaves)
    Cuneiform                 6.11x   0/406
    alchemy_full_bond         4.97x   0/102
    BZR_bond                  2.37x   1/84
    QM9_bond                  2.03x   54/156

At the bound itself the rate is 50-75%, but a pattern at the bound has no extensions, so
pruning it saves nothing: the subtree it would cut is empty.

WHY, and it is not a matter of tuning. The slack term (K-k)*w_max grows linearly in the
remaining depth, and so does the objective's own value, so the ratio bound/true-optimum
stays near w_max/mean-weight -- 2x to 6x on these collections. Pruning needs a front member
to exceed the bound, hence to exceed the subtree's true optimum by that factor, and the
front contains that optimum. The bound can only fire once the slack is nearly zero.

A first hypothesis -- that the -size objective blocks pruning, since U(g)'s size component is
the best any extension can reach -- was tested and REFUTED: dropping size and taking the
front over (WSUP, ATW) alone gives byte-identical prunable counts, 106/712 and 54/365.

THE LEVER, MEASURED 2026-08-23, and it is not enough. An instance-aware bound must stay
sound: edges added beyond the first can attach to newly added vertices, so "edges incident to
the embedding" is not a valid restriction. The tightest sound localisations are the host
graph containing the embedding, and the ball of radius K-k around it.

CORRECTION 2026-08-24. The table that stood here was typed from a run whose parameters were
never recorded, and it is NOT reproducible: scripts/measure_bound_locality.py now computes
these quantities from the definitions and reproduces the weight columns exactly (15.000 /
13.569 / 1.11x and so on) but NOT the coverage column, and it also disagrees on mean host
size (7.7 vs 10.3 nodes for QM9_bond), which points at a different graph limit. Two averagings
were tried against the old digits and neither reproduces them, so the original run is declared
untraceable rather than tidied into agreement. The live numbers are the artefact:

    results-invariant/bound_locality_d3.csv

The conclusion is unchanged -- coverage is 80-100% on three collections of four, and the host
bound gains 1.02x to 1.79x against a gap of 2x to 6x -- but the digits below are dead and are
kept only so the correction has something to point at.

    collection          3*w_max   host top-3 (max)   gain   ball coverage at 3 hops
    Cuneiform            15.000             13.569   1.11x                   100.0%   [dead]
    QM9_bond             13.530             13.230   1.02x                    90.3%   [dead]
    BZR_bond             14.996             11.946   1.26x                    33.5%   [dead]
    alchemy_full_bond    12.146              6.773   1.79x                    84.7%   [dead]

The host-level bound gains 1.02x to 1.79x against a gap of 2x to 6x. The ball collapses onto
the host bound on three collections of four, because these host graphs hold 7.7 to 33 nodes
and a ball of radius K-k already covers 85-100% of them. Instance-awareness has nothing to
localise when the host is barely larger than the pattern bound.

DECIDABLE CRITERION, which is what this negative is worth. Before writing any such bound,
compute the ball coverage at radius equal to the size bound. If it is near 1, the bound
cannot beat the global one and no implementation will change that. It costs seconds and
needs no mining. The setting where it would pass is a single large graph -- WIGM, WeFreS and
GraMi all work there, and a K-ball is a tiny fraction of such a host.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params
from wfsmbench.io import load_dataset
from wfsmbench.match import match_database
from wfsmbench.measure.pareto import dominates, objectives, pareto_front
from wfsmbench.preprocess import build_statistics


def probe(dataset: str, limit: int, bound: int, min_support: int, drop_size: bool = False) -> dict:
    """drop_size: treat size as a CONSTRAINT (the bound) instead of a third objective.

    The first run showed pruning only at the size bound, i.e. only at leaves, where there is
    no subtree to save. The reason is structural: U(g)'s size component is -size(g), the best
    any extension can reach, so only a pattern no larger than g can dominate g's subtree --
    and the smallest patterns are on the front by construction. This variant tests the lever
    that observation points at.
    """
    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs[:limit]
    stats = build_statistics(subset)
    params = Params(min_support=min_support, max_nodes=bound, max_edges=bound,
                    size_reference=STUDY_SIZE_REFERENCE)

    started = time.perf_counter()
    pool = mine_exhaustive(subset, stats, params).pool
    mined = time.perf_counter() - started
    def obj(c):
        full = objectives(c.evaluation, c.particle.pattern)
        return full[:2] if drop_size else full
    points = [obj(c) for c in pool]
    front = pareto_front(points)

    w_max = stats.max_edge_weight
    prunable = 0
    per_size: dict[int, list[int]] = {}
    for candidate in pool:
        pattern = candidate.particle.pattern
        k, v = pattern.num_edges(), pattern.num_nodes()
        embeddings = match_database(pattern, subset)
        max_w = max((sum(w) for lists in embeddings.values() for w in lists), default=0.0)
        slack = (bound - k) * w_max
        atw_ub = max_w + slack
        upper = ((candidate.evaluation.support * atw_ub, atw_ub) if drop_size
                 else (candidate.evaluation.support * atw_ub, atw_ub, -float(v + k)))
        cut = any(dominates(f, upper) for f in front)
        prunable += cut
        per_size.setdefault(k, [0, 0])
        per_size[k][0] += cut
        per_size[k][1] += 1

    return {"dataset": dataset, "limit": limit, "bound": bound, "min_support": min_support,
            "pool": len(pool), "front": len(front), "prunable": prunable,
            "per_size": per_size, "mine_seconds": mined}


if __name__ == "__main__":
    for args, drop in [(("Cuneiform", 60, 4, 2), False), (("QM9_bond", 200, 4, 2), False),
                       (("BZR_bond", 200, 4, 2), False), (("alchemy_full_bond", 200, 4, 2), False)]:
        r = probe(*args, drop_size=drop); r["mode"] = "2 objectives (size dropped)" if drop else "3 objectives"
        pct = 100.0 * r["prunable"] / r["pool"] if r["pool"] else 0.0
        print(f"\n[{r['mode']}] {r['dataset']} n={r['limit']} bound={r['bound']} min_support={r['min_support']}"
              f"  (mined in {r['mine_seconds']:.0f}s)")
        print(f"  pool {r['pool']}, front {r['front']}, prunable {r['prunable']} = {pct:.1f}%")
        for k in sorted(r["per_size"]):
            cut, tot = r["per_size"][k]
            print(f"    k={k}: {cut}/{tot} = {100.0*cut/tot:5.1f}%")
        sys.stdout.flush()
