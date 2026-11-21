"""How much evidence each proved statement is actually checked against.

Every statement in the theory names a measurement that would refute it, and the value of
that arrangement is entirely in the DENOMINATOR: "no violations" over two comparable cases
says almost nothing, and over a hundred and forty-four it says a great deal. So the counts
belong beside the statements, and they belong to one implementation -- the checks and the
manuscript table must not each compute them, or the paper can report a breadth the checks
never had.

Loading is separated from counting so the tests can skip on a missing artefact while the
generator refuses on one. Nothing here decides whether a statement holds; that stays in
tests/test_theory_predictions.py, which imports these same loaders.
"""

from __future__ import annotations

import collections
import csv
import glob
from pathlib import Path

# The study defaults, from config.py. Repeated on purpose: if the defaults move, the checks
# must be re-derived rather than quietly follow them. tests/test_verification_defaults.py
# fails the moment the two disagree.
#
# CORRECTED 2026-11-21. SIZE_REFERENCE was 6, the unit-test default of Params, while every
# recorded fitness uses the study anchor STUDY_SIZE_REFERENCE = 7. It scales the per_edge
# objective read from sum-mode rows, so the growth-dichotomy check ran on a per_edge objective
# no run had used. Re-run at 7, the counts are unchanged (14 of 14 cells, all Cuneiform).
ALPHA, BETA, GAMMA, ETA1, ETA2, SIZE_REFERENCE = 1.0, 0.5, 0.1, 0.1, 0.1, 7


def read(results: Path, name: str) -> list[dict]:
    """Rows of one artefact, or an empty list when it is absent."""
    path = results / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frontier_series(results: Path) -> dict[tuple[int, int], list[dict]]:
    by: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for row in read(results, "degeneracy_frontier_cuneiform.csv"):
        by[(int(row["max_size"]), int(row["min_support"]))].append(row)
    return {k: sorted(v, key=lambda r: int(r["edges"])) for k, v in by.items()}


def gamma_series(results: Path) -> dict[tuple, list[tuple[float, int, int]]]:
    by: dict[tuple, list[tuple[float, int, int]]] = collections.defaultdict(list)
    for path in sorted(glob.glob(str(results / "degeneracy_gamma_*.csv"))):
        with open(path, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                key = (r["dataset"], r["graph_limit"], r["max_size"], r["min_support"])
                by[key].append((float(r["gamma"]), int(r["peak_edges"]), int(r["peak_nodes"])))
    return {k: sorted(v) for k, v in by.items()}


def pareto_rows(results: Path) -> list[dict]:
    """Front rows, excluding the partial re-measurements that cover cells twice.

    With the suffixed files included the row count was 148 per selector against 141 distinct
    cells, and the seven extra were exactly those files.
    """
    rows: list[dict] = []
    for path in sorted(glob.glob(str(results / "pareto_size_*.csv"))):
        if path.endswith("_s45.csv") or path.endswith("_s4.csv"):
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            rows += list(csv.DictReader(handle))
    return rows


def selector_cells(results: Path) -> dict:
    """(dataset, mode, size, support) -> budget -> method -> seed -> fitness."""
    cells: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for path in sorted(glob.glob(str(results / "weight_mode_runs_sel*.csv"))):
        with open(path, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                key = (r["dataset"], r["weight_mode"], int(r["max_size"]),
                       int(r["min_support"]))
                cells[key][int(r["budget"])][r["method"]][int(r["seed"])] = float(r["fitness"])
    return cells


def predicted_argmax(series: list[dict], mode: str) -> int | None:
    """The growth dichotomy applied to one configuration, from sum-mode frontier data only.

    per_edge replaces a_hat by a/(k*w_max) = SIZE_REFERENCE * a_hat / k, so its objective is
    SIZE_REFERENCE * G(k)/k minus the same penalty. Nothing here reads a per_edge run, which
    is what makes the per_edge prediction an out-of-sample one.
    """
    best_k, best_value = None, None
    for row in series:
        k, v, g = int(row["edges"]), int(row["nodes"]), float(row["G"])
        core = SIZE_REFERENCE * g / k if mode == "per_edge" else g
        value = core - GAMMA * (ETA1 * v + ETA2 * k)
        if best_value is None or value > best_value:
            best_k, best_value = k, value
    return best_k


# ------------------------------------------------------------------------- denominators

def frontier_rows(results: Path) -> int:
    return len(read(results, "degeneracy_frontier_cuneiform.csv"))


def gamma_steps(results: Path) -> tuple[int, int, int]:
    """(steps compared, series, collections) for the size-dial check."""
    series = gamma_series(results)
    steps = sum(max(len(points) - 1, 0) for points in series.values())
    return steps, len(series), len({key[0] for key in series})


def dichotomy_cells(results: Path, mode: str) -> int:
    """Configurations where a prediction and a measurement can be compared."""
    series = frontier_series(results)
    measured = {(int(r["size_bound"]), int(r["min_support"]))
                for r in read(results, "weight_mode_landscape.csv")
                if r["weight_mode"] == mode}
    return len([k for k in series if k in measured])


def dichotomy_steps(series: list[dict], mode: str) -> list[tuple[float, int]]:
    """(delta G, delta v) between consecutive best patterns of one cell, in the given mode.

    The sequence is the frontier: for each edge count, the pattern with the largest F. per_edge
    is read from the same sum-mode rows, with G scaled by SIZE_REFERENCE / k as in
    predicted_argmax, so nothing here reads a per_edge run either.
    """
    steps = []
    for a, b in zip(series, series[1:]):
        ka, kb = int(a["edges"]), int(b["edges"])
        ga, gb = float(a["G"]), float(b["G"])
        if mode == "per_edge":
            ga, gb = SIZE_REFERENCE * ga / ka, SIZE_REFERENCE * gb / kb
        steps.append((gb - ga, int(b["nodes"]) - int(a["nodes"])))
    return steps


def strict_case_holds(steps: list[tuple[float, int]], mode: str) -> bool:
    """Whether every step meets the strict case of the growth dichotomy for this mode.

    sum: delta G > gamma (eta1 + eta2) with delta v <= 1, the rising case.
    per_edge: delta G < gamma eta2 with delta v >= 0, the falling case.
    """
    if mode == "per_edge":
        return all(dg < GAMMA * ETA2 and dv >= 0 for dg, dv in steps)
    return all(dg > GAMMA * (ETA1 + ETA2) and dv <= 1 for dg, dv in steps)


def dichotomy_counts(results: Path, mode: str) -> dict[str, int]:
    """Counts behind the growth-dichotomy paragraph, over the cells dichotomy_cells counts.

    cells: cells compared; strict: cells where every step meets the strict case of this mode;
    predicted: cells where the frontier prediction gives the measured optimum's edge count;
    at_bound: cells whose measured optimum has the LARGEST edge count of the cell's pool. That is
    the manuscript's definition since 2026-11-20 (ticket 5): it equals K unless no frequent
    pattern has K edges, and a pattern with K vertices but fewer edges than the pool's largest is
    not at the bound, since it can still grow by an edge that closes a cycle.
    """
    series = frontier_series(results)
    measured = {(int(r["size_bound"]), int(r["min_support"])): r
                for r in read(results, "weight_mode_landscape.csv") if r["weight_mode"] == mode}
    cells = [key for key in sorted(series) if key in measured]
    counts = {"cells": len(cells), "strict": 0, "predicted": 0, "at_bound": 0}
    for key in cells:
        row = measured[key]
        counts["strict"] += strict_case_holds(dichotomy_steps(series[key], mode), mode)
        counts["predicted"] += predicted_argmax(series[key], mode) == int(row["optimum_edges"])
        counts["at_bound"] += int(row["optimum_edges"]) == max(int(r["edges"]) for r in series[key])
    return counts


def front_cells(results: Path) -> tuple[int, int]:
    """(cells, collections) behind the dominance-invariance consequence."""
    rows = pareto_rows(results)
    cells = {(r["dataset"], r["size_bound"], r["min_support"]) for r in rows}
    return len(cells), len({r["dataset"] for r in rows})


def racing_cells(results: Path) -> int:
    """Cells carrying per-seed runs at two budgets, which the bound needs to instantiate."""
    usable = 0
    for cell in selector_cells(results).values():
        budgets = sorted(cell)
        if len(budgets) < 2:
            continue
        methods = set(cell[budgets[0]]) & set(cell[budgets[-1]])
        seeds = {s for m in cell[budgets[0]].values() for s in m}
        if len(methods) >= 2 and len(seeds) >= 4:
            usable += 1
    return usable


def sharp_and_plateau(results: Path) -> tuple[int, int]:
    rows = [r for r in read(results, "selector_regret_allpairs.csv")
            if r["seed_split"] == "disjoint"]
    return (sum(1 for r in rows if r["top"] == "sharp"),
            sum(1 for r in rows if r["top"] == "plateau"))
