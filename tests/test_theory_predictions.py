"""Every prediction the theory makes about the results, recomputed from the artefacts.

A proof is an argument; these are the measurements that would refute one. Each check prints
its denominator, because "no violations" over zero comparable cases is not a pass.

The predictions are computed here from the definitions, not read from a column someone
already wrote -- the frontier prediction in particular derives the per-edge outcome from
sum-mode data alone, so agreeing with the per-edge artefact is a real test and not a
restatement.
"""
from __future__ import annotations

import collections
import csv
import glob
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

from wfsmbench.measure import verification as V  # noqa: E402

# The study defaults, from src/wfsmbench/config.py. Duplicated on purpose: if the defaults
# move, these checks must be re-derived rather than silently follow.
ALPHA, BETA, GAMMA, ETA1, ETA2, SIZE_REFERENCE = 1.0, 0.5, 0.1, 0.1, 0.1, 6


def read(name: str) -> list[dict]:
    rows = V.read(RESULTS, name)
    if not rows:
        pytest.skip(f"artefact not present or empty: {name}")
    return rows

def frontier_series() -> dict[tuple[int, int], list[dict]]:
    return V.frontier_series(RESULTS)

def test_v1_the_no_clipping_hypothesis_holds() -> None:
    """The exact factorisation assumes neither clip is active; if one is, the identity lapses."""
    rows = [r for series in frontier_series().values() for r in series]
    assert rows, "no frontier rows to check"
    worst_a = max(float(r["a_hat"]) for r in rows)
    worst_w = max(float(r["a_hat"]) * float(r["s_hat"]) for r in rows)
    assert worst_a < 1.0, f"a_hat reaches {worst_a}; norm() clips and the exact factorisation lapses"
    assert worst_w < 1.0, f"normalised WSUP reaches {worst_w}; norm() clips"
    print(f"\n[no clipping] {len(rows)} rows; largest a_hat {worst_a:.4f}, largest WSUP {worst_w:.4f}")


# ---------------------------------------------- optimal size is monotone in gamma


def gamma_series() -> dict[tuple, list[tuple[float, int, int]]]:
    return V.gamma_series(RESULTS)

def test_v2_the_argmax_is_non_increasing_in_gamma() -> None:
    """The penalty coefficient is a size dial: raising it only moves the answer down."""
    series = gamma_series()
    if not series:
        pytest.skip("no gamma artefacts")
    steps = violations = 0
    for key, rows in series.items():
        for (g1, e1, v1), (g2, e2, v2) in zip(rows, rows[1:]):
            steps += 1
            if e2 > e1 or (v2 + e2) > (v1 + e1):
                violations += 1
                print(f"  gamma {g1}->{g2} at {key}: edges {e1}->{e2}, size {v1+e1}->{v2+e2}")
    assert steps >= 100, f"only {steps} gamma steps available; the check would be thin"
    print(f"\n[monotone in gamma] {steps} steps over {len(series)} series, "
          f"{len({k[0] for k in series})} collections")
    assert violations == 0, f"{violations} of {steps} steps move the argmax upward"


# ---------------------------------------------- the frontier predicts the optimum


def predicted_argmax(series: list[dict], mode: str) -> int:
    return V.predicted_argmax(series, mode)


@pytest.mark.parametrize("mode", ["sum", "per_edge"])
def test_v3_the_frontier_argmax_predicts_the_measured_optimum(mode: str) -> None:
    series = frontier_series()
    measured = {}
    for row in read("weight_mode_landscape.csv"):
        if row["weight_mode"] == mode:
            measured[(int(row["size_bound"]), int(row["min_support"]))] = int(row["optimum_edges"])
    comparable = [k for k in series if k in measured]
    assert len(comparable) >= 10, (
        f"only {len(comparable)} configurations can be compared for mode {mode}; "
        f"the check would be close to vacuous"
    )
    wrong = [(k, predicted_argmax(series[k], mode), measured[k])
             for k in sorted(comparable)
             if predicted_argmax(series[k], mode) != measured[k]]
    print(f"\n[frontier prediction/{mode}] {len(comparable)} configurations compared")
    assert not wrong, f"{len(wrong)} of {len(comparable)} predictions wrong: {wrong[:4]}"


# ---------------------------------------------- where the selectors land on the front


def pareto_rows() -> list[dict]:
    return V.pareto_rows(RESULTS)

def test_v4_only_the_scalarisation_reaches_the_top_of_the_front() -> None:
    """The parameter-free selectors stop short of the front's maximum when the front has an
    interior; the scalarised one goes to it.

    CORRECTED 2026-09-28. This used to assert that ideal and maxmin NEVER reach the front
    maximum, and called that the measurable consequence of Pareto invariance. Both halves
    were wrong. The proposition says dominance, hence the non-dominated set, is unchanged by
    strictly increasing reparameterisation; it says nothing about where a selector lands on
    the set. And the unconditional claim held only because every committed front spanned
    three or more edge levels: re-running the _bond Pareto stage at bound 3, where most
    fronts span two, put ideal and maxmin on the maximum in 60 of 66 rows. With two levels
    there is no interior to stop at, so "not the maximum" can only mean "the minimum".
    The claim is now scoped to fronts with an interior, and the two-level rows are printed
    rather than hidden.
    """
    rows = pareto_rows()
    if not rows:
        pytest.skip("no Pareto artefacts")
    cells = len({(r["dataset"], r["size_bound"], r["support_fraction"]) for r in rows})
    assert cells >= 100, f"only {cells} cells; the separation claim needs breadth"
    top = collections.Counter()
    total = collections.Counter()
    for r in rows:
        levels = len(r["front_edge_range"].split("|"))
        scope = "interior" if levels >= 3 else "two-level"
        total[(scope, r["selector"])] += 1
        top[(scope, r["selector"])] += int(r["picked_edges"] == r["front_max_edges"])
    print(f"\n[front maximum] {cells} cells over {len({r['dataset'] for r in rows})} collections; "
          f"at the front maximum: " +
          ", ".join(f"{s}/{scope} {top[(scope, s)]}/{total[(scope, s)]}"
                    for scope, s in sorted(total)))
    for selector in ("ideal", "maxmin"):
        n = total[("interior", selector)]
        assert n >= 100, (
            f"only {n} {selector} rows on fronts with three or more edge levels; the check "
            f"would be close to vacuous")
        assert top[("interior", selector)] == 0, (
            f"{selector} reaches the maximum of a front with an interior in "
            f"{top[('interior', selector)]} of {n} rows"
        )
    top = collections.Counter({s: top[("interior", s)] + top[("two-level", s)]
                               for _scope, s in total})
    total = collections.Counter({s: total[("interior", s)] + total[("two-level", s)]
                                 for _scope, s in total})
    assert top["weighted"] > total["weighted"] // 2, (
        f"the scalarisation sits at the front maximum in only {top['weighted']} of "
        f"{total['weighted']}; the contrast the corollary predicts is not there"
    )


# ---------------------------------------------- the regret bound of the race


def selector_cells() -> dict:
    cells = V.selector_cells(RESULTS)
    if not cells:
        pytest.skip("per-seed runs are not present; the racing checks need them")
    return cells

def racing_terms(cell: dict) -> tuple | None:
    """(2*epsilon, D, realised regret, transfer loss) for one cell, or None if unusable."""
    import math
    import statistics

    budgets = sorted(cell)
    if len(budgets) < 2:
        return None
    cheap, dear = budgets[0], budgets[-1]
    methods = sorted(set(cell[cheap]) & set(cell[dear]))
    seeds = sorted({s for m in cell[cheap].values() for s in m})
    if len(seeds) < 4 or len(methods) < 2:
        return None
    race = [s for s in seeds if s % 2 == 0]
    if not race:
        return None

    mu = {m: statistics.mean(cell[cheap][m].values()) for m in methods}
    nu = {m: statistics.mean(cell[dear][m].values()) for m in methods}
    mu_race = {m: statistics.mean([cell[cheap][m][s] for s in race if s in cell[cheap][m]])
               for m in methods}
    pick = max(mu_race, key=mu_race.get)

    gains = {m: nu[m] - mu[m] for m in methods}
    spread = max(gains.values()) - min(gains.values())
    values = [v for m in methods for v in cell[cheap][m].values()]
    width = max(values) - min(values)
    epsilon = width * math.sqrt(math.log(2 * len(methods) / 0.05) / (2 * len(race)))

    cheap_best, dear_best = max(mu, key=mu.get), max(nu, key=nu.get)
    return (2 * epsilon, spread, nu[dear_best] - nu[pick], nu[dear_best] - nu[cheap_best])


def test_v5_the_racing_bound_holds() -> None:
    """The racing bound: realised regret must not exceed 2*epsilon + D in any cell."""
    terms = [t for t in (racing_terms(c) for c in selector_cells().values()) if t]
    assert len(terms) >= 8, f"only {len(terms)} usable cells; the check would be thin"
    breaches = [(e, d, r) for e, d, r, _ in terms if r > e + d + 1e-12]
    print(f"\n[racing bound] {len(terms)} cells; mean bound "
          f"{sum(e + d for e, d, _, _ in terms) / len(terms):.4f}, "
          f"mean realised {sum(r for _, _, r, _ in terms) / len(terms):.6f}")
    assert not breaches, f"{len(breaches)} cells exceed the bound: {breaches[:3]}"


def test_v5b_the_top_is_a_plateau_which_is_why_the_bound_is_loose() -> None:
    """The stated reason for the ~1250x looseness, held to the data.

    If the transfer loss were large the bound would be tight and the explanation wrong. It
    is not: paying for the cheap-budget best instead of the dear-budget best costs almost
    nothing, which is the mechanism the manuscript claims.
    """
    terms = [t for t in (racing_terms(c) for c in selector_cells().values()) if t]
    transfer = [t for *_, t in terms]
    print(f"\n[plateau] transfer loss over {len(transfer)} cells: "
          f"mean {sum(transfer)/len(transfer):.5f}, max {max(transfer):.5f}")
    assert max(transfer) < 0.01, (
        f"the largest transfer loss is {max(transfer):.4f}; the plateau explanation for the "
        f"bound's looseness does not hold and the manuscript's mechanism is wrong"
    )


def test_v5c_the_loss_lives_entirely_in_the_sharp_cells() -> None:
    """The open case, tested on real data rather than constructed.

    The earlier headline -- 0.156% mean regret -- was measured on eleven cells, and it turns
    out most of them are cells where nothing is at stake. Scoring every budget pair reaches
    39 combinations with a sharp top, and there the selector does lose.
    """
    path = RESULTS / "selector_regret_allpairs.csv"
    if not path.exists():
        pytest.skip("run scripts/run_selector.py --all-pairs first")
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["seed_split"] == "disjoint"]
    sharp = [float(r["regret_race"]) for r in rows
             if r["top"] == "sharp" and r["regret_race"]]
    plateau = [float(r["regret_race"]) for r in rows
               if r["top"] == "plateau" and r["regret_race"]]
    print(f"\n[sharp and plateau cells] {len(sharp)} sharp cells, {len(plateau)} plateau cells")
    assert len(sharp) >= 20 and len(plateau) >= 10, (
        f"only {len(sharp)} sharp and {len(plateau)} plateau cells; the contrast needs both"
    )
    assert max(plateau) == 0.0, (
        f"a plateau cell shows regret {max(plateau)}; the claim that nothing is at stake "
        f"where methods tie at the top is wrong"
    )
    assert max(sharp) > 1.0, (
        f"the worst sharp-cell regret is {max(sharp)}, so the selector never actually loses "
        f"and the case the manuscript calls untested is not being reached"
    )


def test_v5d_the_worst_losses_come_from_the_fixable_term() -> None:
    """Which term of the racing bound does the damage. This corrects an earlier claim.

    D was described as the irreducible part and, by implication, the one that matters. The
    measurement says the opposite: estimation error at the race budget -- the term that
    shrinks as the number of race seeds grows -- causes the large losses.
    """
    path = RESULTS / "selector_regret_allpairs.csv"
    if not path.exists():
        pytest.skip("run scripts/run_selector.py --all-pairs first")
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["seed_split"] == "disjoint" and r["regret_race"]]
    lost = [r for r in rows if float(r["regret_race"]) > 0.01]
    estimation = [float(r["regret_race"]) for r in lost if r["estimated_right"] == "0"]
    transfer = [float(r["regret_race"]) for r in lost if r["estimated_right"] == "1"]
    assert estimation and transfer, "both failure modes must occur for the split to mean anything"
    print(f"\n[fixable term] {len(lost)} losses: estimation {len(estimation)} (worst "
          f"{max(estimation):.3f}%), transfer {len(transfer)} (worst {max(transfer):.3f}%)")
    assert max(estimation) > max(transfer), (
        f"estimation error peaks at {max(estimation):.3f}% and transfer failure at "
        f"{max(transfer):.3f}%; the claim that the fixable term does the damage is wrong"
    )


def sweep_rows() -> list[dict]:
    path = RESULTS / "selector_seed_sweep.csv"
    if not path.exists():
        pytest.skip("run scripts/run_selector_seed_sweep.py first")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_v5e_more_race_seeds_reaches_the_transfer_floor_and_stops() -> None:
    """The racing bound's statistical term is exhausted well before the seeds run out.

    Regret must fall as the race set grows and must not fall below the regret a perfect
    race would incur, which is the transfer term. If it went below, the floor would be
    computed wrongly and the decomposition would mean nothing.
    """
    rows = sweep_rows()
    def col(name):
        return [float(r[name]) for r in rows if r[name]]
    small, large, floor = col("regret_n2"), col("regret_n10"), col("transfer_only")
    assert len(small) == len(large) == len(floor) >= 40
    mean_small = sum(small) / len(small)
    mean_large = sum(large) / len(large)
    mean_floor = sum(floor) / len(floor)
    print(f"\n[race size] n=2 {mean_small:.3f}%  n=10 {mean_large:.3f}%  floor {mean_floor:.3f}%")
    assert mean_large < mean_small, "more race seeds did not help at all"
    # On the mean only: one race may beat the cheap-budget best in a cell (that cell's
    # estimation share is then negative), but across cells a ten-seed race cannot do better on
    # average than knowing the cheap budget on every seed.
    assert mean_large >= mean_floor - 1e-9, (
        f"mean regret {mean_large:.4f} is below the mean transfer floor {mean_floor:.4f}"
    )


def test_v5f_the_large_losses_are_estimation_not_transfer() -> None:
    """Which term of the racing bound is responsible for each realized loss.

    REFUTED 2026-11-21. This test used to be test_v5f_the_large_losses_are_transfer_not_
    estimation, and it asserted that the estimation share (regret at ten race seeds minus the
    transfer floor) stays below 1% everywhere, so every large loss was transfer. It passed
    only because the floor was read from the race's own seeds: in a 20-seed cell the race at
    ten seeds IS that floor, and its estimation share was zero by construction. Three
    independent reviews found it; the transfer column of selector_regret_allpairs.csv shows
    the two budgets agreeing in all six cells above 1%.

    Corrected statement, read per race from the disjoint-seed rows: every loss above 1%
    comes from a race that missed the cheap-budget best while the two budgets agreed, and
    no loss of a race whose budgets disagreed reaches 1%.
    """
    rows = [r for r in read("selector_regret_allpairs.csv") if r["seed_split"] == "disjoint"]
    assert len(rows) >= 40
    large = [r for r in rows if float(r["regret_race"]) > 1.0]
    transfer_fails = [r for r in rows if r["transfers"] == "0"]
    both_hold = [r for r in rows if r["transfers"] == "1" and r["estimated_right"] == "1"]
    print(f"\n[loss terms] {len(rows)} races; {len(large)} lose more than 1%; "
          f"{len(transfer_fails)} with the budgets disagreeing, worst "
          f"{max(float(r['regret_race']) for r in transfer_fails):.3f}%; "
          f"{len(both_hold)} with both terms holding")
    assert large, "no race loses more than 1%, so there is nothing to attribute"
    assert all(r["transfers"] == "1" and r["estimated_right"] == "0" for r in large), (
        "a loss above 1% comes from a race whose budgets disagree")
    assert all(float(r["regret_race"]) < 1.0 for r in transfer_fails), (
        "a race whose budgets disagree loses 1% or more")
    # By definition: the race picks the cheap-budget best and that is the paying-budget best,
    # so it pays nothing. A failure here means the two columns are not what they say.
    assert all(float(r["regret_race"]) == 0.0 for r in both_hold)


def test_v5g_the_transfer_floor_does_not_reuse_the_race() -> None:
    """The floor must not equal the race it is meant to bound.

    With the floor read from the race seeds, a 20-seed cell had transfer_only equal to its
    regret at ten race seeds, so the decomposition could not see estimation there. The floor
    now reads the cheap budget on every seed of the cell; in some 20-seed cell it must
    therefore differ from the ten-seed race, or the correction has not reached the artefact.
    """
    rows = [r for r in sweep_rows() if r["reach"] == "10"]
    assert rows, "no 20-seed cell in the sweep"
    differing = [r for r in rows if r["regret_n10"] != r["transfer_only"]]
    print(f"\n[floor] {len(differing)} of {len(rows)} twenty-seed cells have a floor that "
          f"differs from the ten-seed race")
    assert differing
