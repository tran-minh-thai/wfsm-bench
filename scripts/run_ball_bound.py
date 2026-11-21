"""T5 of docs/SUPPLEMENTARY_PROTOCOL.md: a real ball bound for front-membership pruning.

The coverage criterion said, from a count that needs no mining, when a ball bound cannot be
tighter than the host-graph bound. This writes the three bounds into a breadth-first
enumerator of frequent patterns and measures what each one cuts, so the criterion can be
checked against an actual pruning run with its threshold fixed in advance (tau = 0.9).

A frequent pattern g with k < K edges is not expanded when a member of the front built so far
dominates the upper vector (supp(g) U, U, -(v + k)), where U is the largest, over the
embeddings phi of g, of A(phi) + rem(phi), and d = K - k:

    global  rem = d w_max
    host    the d largest edge weights of the host graph of phi, edges of phi excluded
    ball    the d largest edge weights among the host edges with both endpoints within
            distance d of the vertices of phi, edges of phi excluded

Every descendant g' of g has supp(g') <= supp(g), each of its embeddings extends an embedding
of g by at most d host edges inside that ball, and ATW is a mean of embedding sums, so the
vector bounds every descendant and a dominated subtree holds no front member. The run
refuses unless the front with pruning equals the front without it, for every bound.
`--inject-unsound` adds a bound that cuts every non-leaf pattern, which is not sound, to show
that this check catches one. (A first version injected rem = 0; on QM9_bond it cut 49 patterns
and happened to lose no front member, so it could not show anything.)

The enumerator is the one of baselines/exhaustive.py with the cut added; evaluations and the
embedding data of a pattern are cached by canonical signature, since they are a function of
the pattern, so the four runs of a dataset match each pattern once. Counts only.

    python scripts/run_ball_bound.py
    python scripts/run_ball_bound.py --inject-unsound
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from networkx.algorithms.isomorphism import GraphMatcher  # noqa: E402

import run_weight_mode as W                                     # noqa: E402
from wfsmbench.baselines._common import as_particle, evaluate, extensions, seed_particles  # noqa: E402
from wfsmbench.core.signature import canonical_signature       # noqa: E402
from wfsmbench.io import load_dataset                          # noqa: E402
from wfsmbench.match.matcher import _edge_match, _node_match    # noqa: E402
from wfsmbench.measure.pareto import dominates, objectives, pareto_front  # noqa: E402
from wfsmbench.preprocess import build_statistics              # noqa: E402
from wfsmbench.pso.edits import is_valid                       # noqa: E402

OUT = ROOT / "results-invariant" / "revision-2026-11-21"
DATASETS = ["Cuneiform", "QM9_bond", "BZR_bond", "alchemy_full_bond"]
LIMIT, SIZE, SUPPORT = 200, 4, 2
TAU, FACTOR = 0.9, 1.05
COVERAGE = ROOT / "results-invariant" / "bound_locality_d3.csv"


class Host:
    """What the bounds need from one host graph: adjacency and its edges by weight."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.adjacent: dict[int, list[int]] = {u: [] for u in graph.node_labels}
        for (u, v) in graph.edges:
            self.adjacent[u].append(v)
            self.adjacent[v].append(u)
        self.by_weight = sorted(((w, e) for e, (_l, w) in graph.edges.items()), reverse=True)

    def top(self, d: int, excluded: frozenset, inside: set | None = None) -> float:
        total, taken = 0.0, 0
        for w, (u, v) in self.by_weight:
            if taken == d:
                break
            if (u, v) in excluded or (inside is not None and (u not in inside or v not in inside)):
                continue
            total += w
            taken += 1
        return total

    def ball(self, start: frozenset, d: int) -> set:
        seen = set(start)
        frontier = deque((u, 0) for u in start)
        while frontier:
            u, depth = frontier.popleft()
            if depth == d:
                continue
            for v in self.adjacent[u]:
                if v not in seen:
                    seen.add(v)
                    frontier.append((v, depth + 1))
        return seen


def embeddings(pattern, host: Host):
    """(sum of matched weights, host nodes, host edges) per distinct embedding, deduplicated
    exactly as wfsmbench.match.matcher.embedding_weights deduplicates."""
    if pattern.num_nodes() > host.graph.num_nodes():
        return []
    matcher = GraphMatcher(host.graph.to_networkx(), pattern.to_networkx(),
                           node_match=_node_match, edge_match=_edge_match)
    seen, out = set(), []
    for mapping in matcher.subgraph_monomorphisms_iter():
        p2h = {pn: hn for hn, pn in mapping.items()}
        edges, total = [], 0.0
        for (pu, pv), _label in pattern.edge_items():
            hu, hv = p2h[pu], p2h[pv]
            edges.append((hu, hv) if hu < hv else (hv, hu))
            total += host.graph.weight(hu, hv)
        occurrence = (frozenset(mapping), frozenset(edges))
        if occurrence in seen:
            continue
        seen.add(occurrence)
        out.append((total, frozenset(mapping), frozenset(edges)))
    return out


def upper_atw(pattern, hosts: list[Host], w_max: float, d: int) -> dict[str, float]:
    """U for each remainder: the largest A(phi) + rem(phi) over every embedding phi."""
    best = {"global": float("-inf"), "host": float("-inf"), "ball": float("-inf"),
            "unsound": float("-inf")}
    for host in hosts:
        for total, nodes, edges in embeddings(pattern, host):
            best["global"] = max(best["global"], total + d * w_max)
            best["host"] = max(best["host"], total + host.top(d, edges))
            best["ball"] = max(best["ball"], total + host.top(d, edges, host.ball(nodes, d)))
            best["unsound"] = float("-inf")   # cuts every non-leaf pattern: not sound
    return best


class Front:
    """The non-dominated set of the points added so far."""

    def __init__(self) -> None:
        self.points: list[tuple] = []

    def add(self, point: tuple) -> None:
        if any(dominates(p, point) or p == point for p in self.points):
            return
        self.points = [p for p in self.points if not dominates(point, p)] + [point]

    def dominates(self, point: tuple) -> bool:
        return any(dominates(p, point) for p in self.points)


def enumerate_front(hosts_raw, stats, params, bound: str | None, cache: dict, uppers: dict,
                    hosts: list[Host], bound_every: bool = False) -> dict:
    """baselines/exhaustive.py::mine_exhaustive with the cut added. Returns the counts."""
    frequent = set(stats.frequent_edge_templates(params.min_support))
    seen: set[str] = set()
    queue: list = []
    front = Front()
    points: list[tuple] = []
    counts = {"evaluations": 0, "cut_patterns": 0, "cut_extensions": 0}

    def consider(pattern) -> None:
        signature = canonical_signature(pattern)
        if signature in seen:
            return
        seen.add(signature)
        particle = as_particle(pattern)
        if not is_valid(particle, frequent, params):
            return
        counts["evaluations"] += 1
        if signature not in cache:
            candidate = evaluate(particle, hosts_raw, stats, params)
            cache[signature] = (candidate.evaluation.support,
                                objectives(candidate.evaluation, pattern))
        support, point = cache[signature]
        if support >= params.min_support:
            points.append(point)
            front.add(point)
            queue.append(pattern)

    for particle in seed_particles(sorted(frequent)):
        consider(particle.pattern)
    index = 0
    while index < len(queue):
        pattern = queue[index]
        index += 1
        k = pattern.num_edges()
        if (bound is not None or bound_every) and k < params.max_edges:
            signature = canonical_signature(pattern)
            if signature not in uppers:
                uppers[signature] = upper_atw(pattern, hosts, stats.max_edge_weight,
                                              params.max_edges - k)
            support, _point = cache[signature]
            u = uppers[signature][bound] if bound is not None else None
            upper = (support * u, u, -float(pattern.num_nodes() + k)) if u is not None else None
            if upper is not None and front.dominates(upper):
                counts["cut_patterns"] += 1
                counts["cut_extensions"] += len(extensions(pattern, frequent))
                continue
        for child in extensions(pattern, frequent):
            consider(child)
    counts["pool"] = len(points)
    counts["front"] = sorted(pareto_front(points))
    return counts


def coverage() -> dict[str, float]:
    with COVERAGE.open(newline="", encoding="utf-8") as handle:
        return {r["dataset"]: float(r["ball_coverage_pooled"]) for r in csv.DictReader(handle)
                if r["depth"] == "3"}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inject-unsound", action="store_true",
                    help="also run an unsound bound and require the soundness check to fail")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args(argv)
    bounds = ["global", "host", "ball"] + (["unsound"] if args.inject_unsound else [])
    cov = coverage()
    rows, criterion = [], []
    for dataset in args.datasets:
        graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
        subset = graphs[:LIMIT]
        stats = build_statistics(subset)
        params = W.params_for(SIZE, 240, "sum", SUPPORT)
        hosts = [Host(g) for g in subset]
        cache, uppers = {}, {}
        # The unpruned run also bounds every non-leaf frequent pattern, so the ratio of the host
        # bound to the ball bound is taken over all of them, not over what a pruned run reaches.
        reference = enumerate_front(subset, stats, params, None, cache, uppers, hosts,
                                    bound_every=True)
        ratios, global_ratios = [], []
        by_bound = {}
        for bound in bounds:
            result = enumerate_front(subset, stats, params, bound, cache, uppers, hosts)
            result["front_equal"] = result["front"] == reference["front"]
            by_bound[bound] = result
            print(f"  {dataset:<18} {bound:<8} evaluations {result['evaluations']:>6} "
                  f"(of {reference['evaluations']}) cut {result['cut_patterns']:>5} patterns, "
                  f"{result['cut_extensions']:>6} extensions; front equal {result['front_equal']}",
                  flush=True)
        for values in uppers.values():
            if values["ball"] > 0:
                ratios.append(values["host"] / values["ball"])
                global_ratios.append(values["global"] / values["host"])
        for bound, result in by_bound.items():
            rows.append({"dataset": dataset, "graph_limit": LIMIT, "max_size": SIZE,
                         "min_support": SUPPORT, "bound": bound, "pool": reference["pool"],
                         "evaluations_unpruned": reference["evaluations"],
                         "evaluations": result["evaluations"],
                         "cut_patterns": result["cut_patterns"],
                         "cut_extensions": result["cut_extensions"],
                         "front_size": len(reference["front"]),
                         "front_equal": int(result["front_equal"]),
                         "patterns_bounded": len(uppers),
                         "host_over_ball_mean": f"{statistics.mean(ratios):.6f}",
                         "host_over_ball_p10": f"{percentile(ratios, 0.10):.6f}",
                         "global_over_host_mean": f"{statistics.mean(global_ratios):.6f}"})
        predicted = cov[dataset] >= TAU
        measured = by_bound["ball"]["cut_patterns"] <= FACTOR * by_bound["host"]["cut_patterns"]
        criterion.append({"dataset": dataset, "coverage_d3": f"{cov[dataset]:.6f}", "tau": TAU,
                          "predicted_no_tighter": int(predicted),
                          "cut_host": by_bound["host"]["cut_patterns"],
                          "cut_ball": by_bound["ball"]["cut_patterns"],
                          "measured_no_tighter": int(measured),
                          "correct": int(predicted == measured)})

    sound = [r for r in rows if r["bound"] != "unsound"]
    broken = [r for r in sound if not r["front_equal"]]
    if broken:
        raise SystemExit(f"[refused] pruning changed the front: {[(r['dataset'], r['bound']) for r in broken]}")
    if args.inject_unsound:
        caught = [r for r in rows if r["bound"] == "unsound" and not r["front_equal"]]
        print(f"[inject] the unsound bound changed the front in {len(caught)} of "
              f"{len(args.datasets)} dataset(s)")
        if not caught:
            raise SystemExit("[refused] the soundness check did not catch an unsound bound")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("ball_bound_pruning.csv", rows), ("ball_bound_criterion.csv", criterion)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    import networkx
    import numpy
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    (OUT / "ball_bound.provenance.json").write_text(json.dumps({
        "command": ["python", "scripts/run_ball_bound.py", *argv], "commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": numpy.__version__,
        "networkx": networkx.__version__, "platform": platform.platform(),
        "tau": TAU, "factor": FACTOR}, indent=1) + "\n", encoding="utf-8", newline="")
    for r in criterion:
        print(f"  {r['dataset']:<18} coverage {float(r['coverage_d3']):.3f} predicted no-tighter "
              f"{r['predicted_no_tighter']} cut host {r['cut_host']} ball {r['cut_ball']} "
              f"measured no-tighter {r['measured_no_tighter']} correct {r['correct']}")
    print("[done] ball_bound_pruning.csv, ball_bound_criterion.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
