r"""Why the objective is degenerate in pattern size, and what cannot fix it.

The sweep in run_weight_mode.py established *that* F is monotone in size at every support
threshold the literature uses, so greedy expansion is optimal by construction and the search
comparison is measuring nothing. This script establishes *why*, exactly, and rules out the
obvious repair.

**The factorization.** While neither norm() clips -- checked, not assumed -- F separates into
a growth term and a linear cost:

    F(g) = a_hat * (alpha * s_hat + beta)  -  gamma * (eta1*|V| + eta2*|E|)
           \_________ G ___________/

    a_hat = ATW(g) / (size_reference * W_max)      s_hat = supp(g) / N

So along the frontier Phi(k) = max{F : |E| = k}, the shape is a race between G(k) and a cost
that is linear in the pattern's size. The frontier turns over at the first k where

    G(k+1) - G(k)  <  gamma * (eta1 * dv + eta2),      dv = |V(k+1)| - |V(k)| in {0, 1}

**Two costs, not one.** dv is what makes this exact rather than approximate. An edge that
brings a new node costs gamma*(eta1+eta2); an edge that closes a cycle between nodes already
present costs only gamma*eta2, half as much at the default coefficients.

That also means the frontier is not always unimodal, so "the first k where the condition
fails" is not the peak. Two things are therefore reported, and both are exact rather than
approximate: whether the condition predicts the *direction* of every single step, and whether
the factorization reproduces the measured F well enough that its argmax is the measured peak.
The multimodality is not a defect of the model -- it is what the model predicts, and it
explains the shape that looked anomalous in the sweep: a frontier that falls and then rises
again at the size bound is one where dG sits between the two costs, so tree steps lose and
cycle-closing steps win.

**What it rules out.** gamma, eta1 and eta2 appear only in the linear term. They tilt a line;
they cannot change the shape of G, and the degeneracy is in G. The gamma sweep confirms it on
the objective itself rather than through the model: on Cuneiform at a 3.3% threshold the peak
sits at the size bound for gamma <= 0.2, jumps to the floor at gamma >= 0.4, and is interior
only around 0.3 -- where the optimal fitness is *negative*, the penalty having consumed the
reward. An interior optimum bought that way is an artefact of subtracting a steep enough line,
not a trade-off between support and weight.

The repair therefore has to change G: a_hat grows close to linearly in k because ATW sums the
matched edge weights, while s_hat falls slowly at a slack threshold. The per_edge mode flattens
a_hat and inverts the degeneracy instead of removing it (optimum at a single edge, measured in
every one of 21 configurations). What is needed is an aggregation under which a_hat saturates.

    python scripts/run_degeneracy.py                       # sizes 4,5 -- a few minutes
    python scripts/run_degeneracy.py --sizes 6 --supports 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.baselines import mine_exhaustive                # noqa: E402
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params      # noqa: E402
from wfsmbench.io import load_dataset                          # noqa: E402
from wfsmbench.match import match_database                      # noqa: E402
from wfsmbench.measure import discounted_weight                 # noqa: E402
from wfsmbench.measure.fitness import fitness, norm, penalty    # noqa: E402
from wfsmbench.preprocess import build_statistics              # noqa: E402

RESULTS = ROOT / "results"
DATASET = "Cuneiform"
LIMIT = 60
# rho is calibrated per dataset, and the criterion is stability rather than the mere
# presence of an interior peak: a peak that moves when the size bound moves belongs to
# where the search was cut off, not to the objective. See config.rho.
# Extended below 0.3 after two collections reached the old floor still unstable. Their
# peak was not stuck at the bound -- it sat one edge below it and moved with it -- and
# alchemy's trace shows a peak walking down through smaller interior values while staying
# stable rather than dropping to the floor, so a heavier discount is worth trying before
# concluding the family fails on them.
#
# The tail is geometric rather than linear: what matters is the ratio between successive
# edge contributions, so equal steps in rho are unequal steps in the thing being varied,
# and a linear tail would crowd the region where nothing changes.
RHOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
SUPPORTS = [2, 6, 12, 18, 30, 36, 42]
SIZES = [4, 5]
GAMMAS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.80, 1.00]

FRONTIER_HEADER = ("dataset,graph_limit,max_size,min_support,edges,nodes,support,atw,"
                   "s_hat,a_hat,G,delta_G,delta_v,step_cost,grows,"
                   "fitness_measured,fitness_modelled,clipped")
GAMMA_HEADER = ("dataset,graph_limit,max_size,min_support,gamma,peak_edges,peak_nodes,"
                "peak_shape,peak_fitness")
RHO_HEADER = "dataset,graph_limit,min_support,rho,peak_edges_by_size,peak_fitness_by_size,stable"


def frontier(pool, stats, params):
    """Per edge count, the pattern with the highest F, and what it is made of."""
    best = {}
    for c in pool:
        pattern = c.particle.pattern
        k = pattern.num_edges()
        f = fitness(c.evaluation, pattern.num_nodes(), k, stats, params)
        if k not in best or f > best[k][0]:
            best[k] = (f, pattern.num_nodes(), c.evaluation.support, c.evaluation.atw)
    return best


def shape_of(peak: int, size: int) -> str:
    if peak >= size:
        return "at-the-bound"
    return "interior" if peak > 1 else "at-the-floor"


def refuse_if_other_dataset(path: Path, rows: list[str]) -> None:
    """Refuse to replace a sweep over one dataset with a sweep over another.

    Without --suffix every run writes the same unsuffixed name, so a second
    dataset overwrites the first and leaves a file whose name says nothing about
    what is inside it. That is not hypothetical: a Cuneiform gamma sweep was
    replaced by a QM9 one, the file kept the name the Cuneiform run had written,
    and nothing revealed the loss until the bytes were read. The first two
    columns of every sweep are dataset and graph count, which is enough to tell
    a replacement apart from a re-run.
    """
    if not path.exists() or len(rows) < 2:
        return
    existing = path.read_text(encoding="utf-8").splitlines()
    if len(existing) < 2:
        return
    before = tuple(existing[1].split(",")[:2])
    after = tuple(rows[1].split(",")[:2])
    if before != after:
        raise SystemExit(
            f"[refused] {path.name} holds {before[0]} n={before[1]}; this run writes "
            f"{after[0]} n={after[1]}. Overwriting would destroy the first while "
            f"leaving its file name in place. Pass --suffix to keep them apart."
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", help="comma-separated max sizes")
    ap.add_argument("--supports", help="comma-separated absolute thresholds")
    ap.add_argument("--dataset", default=DATASET, help="dataset JSON stem in data/")
    ap.add_argument("--limit", type=int, default=LIMIT, help="how many graphs")
    # Required, not defaulted. A default writes one shared name, and a second
    # dataset then replaces the first while leaving that name in place -- which is
    # how a Cuneiform gamma sweep was lost. refuse_if_other_dataset catches the
    # collision after the fact; requiring the suffix means it cannot arise.
    ap.add_argument("--suffix", required=True,
                    help="appended to the output filenames. Without it a second dataset "
                         "overwrites the first's results, which is how a QM9 calibration "
                         "survived only in its log.")
    ap.add_argument("--calibrate-rho", action="store_true",
                    help="sweep rho and report which values put the optimum inside the "
                         "size range at EVERY bound, which is what calibration means")
    args = ap.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES
    supports = [int(s) for s in args.supports.split(",")] if args.supports else SUPPORTS
    dataset, limit = args.dataset, args.limit

    graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
    subset = graphs[:limit]
    stats = build_statistics(subset)
    r_w = STUDY_SIZE_REFERENCE * stats.max_edge_weight

    RESULTS.mkdir(parents=True, exist_ok=True)
    frontier_rows = [FRONTIER_HEADER]
    gamma_rows = [GAMMA_HEADER]
    rho_rows = [RHO_HEADER]
    # Calibration needs more than one bound to compare, and one threshold is enough.
    calibration_sizes = sizes if len(sizes) > 1 else [sizes[0], sizes[0] + 1]
    rho_support = supports[0]

    print(f"{dataset} n={limit}  R={STUDY_SIZE_REFERENCE}  "
          f"W_max={stats.max_edge_weight:.2f}", flush=True)
    agree = total = 0

    for size in sizes:
        for support in supports:
            params = Params(max_nodes=size, max_edges=size, min_support=support,
                            size_reference=STUDY_SIZE_REFERENCE)
            pool = mine_exhaustive(subset, stats, params).pool
            if not pool:
                print(f"\nsize {size} supp {support}: pool empty", flush=True)
                continue
            front = frontier(pool, stats, params)
            ks = sorted(front)
            print(f"\nsize {size}  supp {support} ({support / limit:.0%})  pool {len(pool)}",
                  flush=True)
            print(f"  {'k':>2} {'|V|':>3} {'F':>9} {'F model':>9} {'G':>8} {'dG':>8} "
                  f"{'dv':>3} {'cost':>7}  grows", flush=True)

            previous = None
            steps_right = steps_total = 0
            modelled_by_k = {}
            for k in ks:
                f, v, s, a = front[k]
                s_hat, a_hat = s / limit, a / r_w
                g = a_hat * (params.alpha * s_hat + params.beta)
                modelled = g - params.gamma * (params.eta1 * v + params.eta2 * k)
                # The factorization holds only where neither norm() saturates; say so rather
                # than let a clipped row quietly break the identity.
                clipped = int(a > stats.atw_reference(params.size_reference)
                              or s * a > stats.wsup_reference(params.size_reference))
                if previous is None:
                    dg = dv = cost = None
                    grows = ""
                else:
                    dg = g - previous[0]
                    dv = v - previous[1]
                    cost = params.gamma * (params.eta1 * dv + params.eta2)
                    grows = "yes" if dg >= cost else "no"
                    # The criterion is a claim about this step, not about the peak.
                    steps_total += 1
                    steps_right += (dg >= cost) == (f >= front[k - 1][0])
                print(f"  {k:>2} {v:>3} {f:>9.4f} {modelled:>9.4f} {g:>8.4f} "
                      f"{'' if dg is None else f'{dg:+.4f}':>8} "
                      f"{'' if dv is None else dv:>3} "
                      f"{'' if cost is None else f'{cost:.4f}':>7}  {grows}", flush=True)
                frontier_rows.append(
                    f"{dataset},{limit},{size},{support},{k},{v},{s},{a:.6f},"
                    f"{s_hat:.6f},{a_hat:.6f},{g:.6f},"
                    f"{'' if dg is None else f'{dg:.6f}'},"
                    f"{'' if dv is None else dv},"
                    f"{'' if cost is None else f'{cost:.6f}'},{grows},"
                    f"{f:.6f},{modelled:.6f},{clipped}")
                modelled_by_k[k] = modelled
                previous = (g, v)

            predicted = max(modelled_by_k, key=lambda k: modelled_by_k[k])
            measured = max(front, key=lambda k: front[k][0])
            worst = max(abs(modelled_by_k[k] - front[k][0]) for k in ks)
            total += 1
            agree += predicted == measured
            print(f"  -> peak from the model k={predicted}, measured k={measured}   "
                  f"{'AGREE' if predicted == measured else 'DISAGREE'}; "
                  f"step directions {steps_right}/{steps_total}; "
                  f"largest |model - measured| = {worst:.2e}", flush=True)

            # Can the penalty coefficients buy an interior optimum? Re-score the whole pool
            # at each gamma rather than trusting the model to answer its own question.
            for gamma in GAMMAS:
                tuned = Params(max_nodes=size, max_edges=size, min_support=support,
                               size_reference=STUDY_SIZE_REFERENCE, gamma=gamma)
                tuned_front = frontier(pool, stats, tuned)
                peak = max(tuned_front, key=lambda k: tuned_front[k][0])
                gamma_rows.append(
                    f"{dataset},{limit},{size},{support},{gamma},{peak},"
                    f"{tuned_front[peak][1]},{shape_of(peak, size)},"
                    f"{tuned_front[peak][0]:.6f}")

    print(f"\nturn-over condition agrees with the measured peak in {agree}/{total} "
          f"configurations", flush=True)

    # --- rho calibration -------------------------------------------------------
    if args.calibrate_rho:
        print(f"\n=== calibrating rho on {dataset} (n={limit}) ===", flush=True)
        print("Usable only if the optimum is interior at EVERY bound and lands on the same\n"
              "pattern each time: a peak that moves with the bound belongs to where the search\n"
              "was cut off, not to the objective.", flush=True)
        # The pool does not depend on rho -- membership is decided by support -- and neither
        # do the matched edge weights. Only the aggregation does. So each size is enumerated
        # and matched once, and every rho is computed from the same weights, instead of
        # re-running the expensive half eight times.
        per_size = {}
        for size in calibration_sizes:
            base = Params(max_nodes=size, max_edges=size, min_support=rho_support,
                          size_reference=STUDY_SIZE_REFERENCE)
            started = time.perf_counter()
            pool = mine_exhaustive(subset, stats, base).pool
            weights = [(c, [w for ws in match_database(c.particle.pattern, subset).values()
                            for w in ws]) for c in pool]
            per_size[size] = weights
            print(f"  size {size}: {len(pool)} patterns matched once "
                  f"[{time.perf_counter() - started:.0f}s]", flush=True)

        print("  rho  " + "".join(f"size {s:<14}" for s in calibration_sizes), flush=True)
        for rho in RHOS:
            aggregate = discounted_weight(rho)
            atw_ref = stats.atw_reference_discounted(STUDY_SIZE_REFERENCE, rho)
            wsup_ref = limit * atw_ref
            cells, peaks, values = [], [], []
            for size in calibration_sizes:
                params = Params(max_nodes=size, max_edges=size, min_support=rho_support,
                                size_reference=STUDY_SIZE_REFERENCE,
                                weight_mode="discounted", rho=rho)
                front = {}
                for candidate, embeddings in per_size[size]:
                    if not embeddings:
                        continue
                    pattern = candidate.particle.pattern
                    k = pattern.num_edges()
                    atw = sum(aggregate(w) for w in embeddings) / len(embeddings)
                    supp = candidate.evaluation.support
                    value = (params.alpha * norm(supp * atw, wsup_ref)
                             + params.beta * norm(atw, atw_ref)
                             - params.gamma * penalty(pattern.num_nodes(), k, params))
                    if k not in front or value > front[k]:
                        front[k] = value
                peak = max(front, key=lambda k: front[k])
                peaks.append(peak)
                values.append(round(front[peak], 6))
                cells.append(f"k={peak} {shape_of(peak, size):<14}")
            stable = (len(set(peaks)) == 1 and len(set(values)) == 1
                      and 1 < peaks[0] < min(calibration_sizes))
            # %g, not %.1f: the extended tail holds 0.25 and 0.05, which a single
            # decimal renders as 0.2 and 0.1 -- two rows of the table silently claiming
            # to be values that are also in it.
            print(f"  {rho:>4g}  " + "".join(cells)
                  + ("  <- STABLE" if stable else ""), flush=True)
            rho_rows.append(f"{dataset},{limit},{rho_support},{rho},"
                            f"{'|'.join(str(v) for v in peaks)},"
                            f"{'|'.join(str(v) for v in values)},{int(stable)}")

    outputs = [("degeneracy_frontier.csv", frontier_rows),
               ("degeneracy_gamma.csv", gamma_rows)]
    if args.calibrate_rho:
        outputs.append(("degeneracy_rho.csv", rho_rows))
    targets = [(RESULTS / name.replace(".csv", f"{args.suffix}.csv"), rows)
               for name, rows in outputs]
    # Check every target before writing any of them, so a refusal leaves the
    # directory as it was rather than half-replaced.
    for path, rows in targets:
        refuse_if_other_dataset(path, rows)
    for path, rows in targets:
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"[done] {path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
