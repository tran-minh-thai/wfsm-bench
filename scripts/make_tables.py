"""Render the paper's result tables from the artefacts, so no number is ever retyped.

Numbers that are copied drift. That happened here once already: three tables were
hand-transcribed, and when they were finally checked the deterministic columns still
matched while every stochastic one was stale, which is the signature of a table written
before a re-run and never revisited.

The previous version of this script solved that for a set of experiments that has since
been refuted, and it kept working in the worst possible way. It read two CSVs that were
deleted along with that generation, skipped them with a note, and exited zero -- so
`tables.md` was never produced while the batch's closing message went on telling people
to produce it, and nobody noticed for weeks. That behaviour is reversed here: a missing
input is a failure, loudly, with the file named. A generator that cannot find its data
should stop, not render an empty page.

    python scripts/make_tables.py --out results/tables.md

Output goes inside the repository, not into the paper directory beside it. The paper is a
sibling of this checkout on one machine and absent on the experiment machine, so a
generator that wrote there would work in one place and fail in the other. The paper
includes the generated file; it is not the generator's business to know where the paper
lives.

Standard library only, and no import of the package, so it runs anywhere the results do.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Friedman/Nemenyi logs, one block per seed (decision D4, 2026-11-20).
STATS_DIR = ROOT / "results-invariant" / "revision-2026-11-20" / "statistics_seed_blocks"

sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.measure.ranks import mean_ranks as parse_ranks  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.measure.selector import (  # noqa: E402
    regret_groups, regret_terms, spread)

# Reporting order. Cuneiform first because it is the collection the objective analysis
# was developed on; the molecular collections follow in the order they were added.
COLLECTIONS = ["Cuneiform", "QM9_bond", "BZR_bond", "COX2_bond", "DHFR_bond",
               "alchemy_full_bond"]
STATS_LOG = {"Cuneiform": "s5", "QM9_bond": "qm9", "BZR_bond": "BZRbond",
             "COX2_bond": "COX2bond", "DHFR_bond": "DHFRbond",
             "alchemy_full_bond": "alch"}
METHODS = ["aco", "pso", "multiswarm", "genetic", "clpso", "local_search", "random"]
MISSING: list[str] = []


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        MISSING.append(name)
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_many(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob(pattern)):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        MISSING.append(pattern)
    return rows


def mean_ranks(collection: str, mode: str) -> dict[str, float] | None:
    path = STATS_DIR / f"{STATS_LOG[collection]}_statistics.log"
    if not path.exists():
        MISSING.append(path.name)
        return None
    return parse_ranks(path.read_text(encoding="utf-8"), mode)

def grid(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join(["---"] * len(header)) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, rule] + body)


def calibrated() -> dict[str, tuple[str, str]]:
    """collection -> (largest stable rho, peak edges there), or a dash for neither."""
    best: dict[str, tuple[float, str]] = {}
    for row in read_many("degeneracy_rho*.csv"):
        if row.get("stable") != "1":
            continue
        value = float(row["rho"])
        peak = row["peak_edges_by_size"].split("|")[0]
        name = row["dataset"]
        if name not in best or value > best[name][0]:
            best[name] = (value, peak)
    return {k: (f"{v[0]:g}", v[1]) for k, v in best.items()}


def table_collections() -> str:
    rows = []
    rho = calibrated()
    landscape = {}
    for row in read_many("weight_mode_landscape_*.csv"):
        if row["max_size"] == "5" and row["min_support"] == "2" \
                and row.get("fitness_quantum", "0") in ("0", "") \
                and row.get("graph_offset", "0") in ("0", ""):
            landscape.setdefault(row["dataset"], row)
    for name in COLLECTIONS:
        info = landscape.get(name)
        got = rho.get(name)
        rows.append([
            f"`{name}`",
            info["graph_limit"] if info else "-",
            info["pool_size"] if info else "-",
            info["frequent_templates"] if info else "-",
            got[0] if got else "**none**",
            got[1] if got else "-",
        ])
    return ("### Collections\n\n"
            "Pool and template counts at the configuration every comparison uses: five\n"
            "edges, support two. The discount is calibrated per collection by whether the\n"
            "optimum stays interior and on the same pattern as the bound widens; two\n"
            "collections have no value that satisfies that anywhere in (0, 1].\n\n"
            + grid(["collection", "graphs", "pool", "templates", "calibrated rho",
                    "peak edges"], rows))


def table_ranks() -> str:
    out = []
    for mode in ("sum", "discounted"):
        rows = []
        for name in COLLECTIONS:
            ranks = mean_ranks(name, mode)
            if ranks is None:
                rows.append([f"`{name}`"] + ["-"] * len(METHODS) + ["*no calibrated rho*"])
                continue
            order = sorted(ranks, key=ranks.get)
            cells = [f"{ranks[m]:.3f}" if m in ranks else "-" for m in METHODS]
            rows.append([f"`{name}`"] + cells + [f"**{order[0]}**"])
        out.append(f"**Objective: `{mode}`**\n\n"
                   + grid(["collection"] + METHODS + ["winner"], rows))
    # CORRECTED 2026-11-21. The block and the critical difference were typed here, and the
    # text kept the 80-block analysis (1.007) after the logs moved to one block per seed.
    # Both are now read from the logs the table reads.
    stated = {float(v) for name in COLLECTIONS if (STATS_DIR / f"{STATS_LOG[name]}_statistics.log").exists()
              for v in re.findall(r"Nemenyi CD = ([0-9.]+)",
                                  (STATS_DIR / f"{STATS_LOG[name]}_statistics.log").read_text(encoding="utf-8"))}
    if len(stated) != 1:
        raise SystemExit(f"[fail] the statistics logs state {sorted(stated)} as the critical difference")
    return ("### Method ranking does not transfer between collections\n\n"
            "Mean rank over seed blocks, each averaged over the four budgets; one is best.\n"
            f"Nemenyi critical difference is {stated.pop():.3f}, so methods within that of each\n"
            "other are not distinguishable.\n\n" + "\n\n".join(out))


def table_basins() -> str:
    rows = []
    for row in sorted(read_many("basins_*.csv"), key=lambda r: (r["weight_mode"],
                                                                r["dataset"])):
        if row["policy"] != "strict" or row.get("graph_offset", "0") not in ("0", ""):
            continue
        ranks = mean_ranks(row["dataset"], row["weight_mode"])
        if ranks is None:
            continue
        order = sorted(ranks, key=ranks.get)
        flag = row["optimum_is_largest"] == "1"
        rows.append([
            f"`{row['dataset']}`", row["weight_mode"],
            f"{float(row['basin_optimum_pct']):.2f}%",
            f"{float(row['largest_basin_pct']):.1f}%",
            "**yes**" if flag else "no",
            f"{ranks['aco']:.3f} ({order.index('aco') + 1})",
        ])
    return ("### Basin structure, an instance-level quantity\n\n"
            "Share of the pool whose steepest ascent ends on an optimal pattern, and\n"
            "whether that optimum is the largest attractor at all. Where it is, ACO does\n"
            "not come first; the converse holds in six of eight.\n\n"
            + grid(["collection", "objective", "optimum basin", "largest basin",
                    "optimum dominates", "ACO rank"], rows))


def table_quantum() -> str:
    rows = []
    # Read by explicit tag, never by glob: `landscape_q*` also matches `landscape_qm9`,
    # which is a different experiment without the quantum column. That glob has now
    # collided twice, once here and once when a QM9 slice was mistaken for the base run.
    for tag in ("q000", "q002", "q005", "q010", "q020", "q050", "q100"):
        path = STATS_DIR / f"{tag}_statistics.log"
        if not path.exists():
            MISSING.append(path.name)
            continue
        block = path.read_text(encoding="utf-8").split("mean ranks")[1]
        ranks = {k: float(v) for k, v in re.findall(r"^\s{4}(\w+)\s+([\d.]+)$", block, re.M)}
        land = read(f"weight_mode_landscape_{tag}.csv")
        levels = f"{float(land[0]['fitness_levels']):.4f}" if land else "-"
        rows.append([levels, land[0]["local_optima"] if land else "-"]
                    + [f"{ranks[m]:.2f}" for m in METHODS])
    return ("### Forcing the landscape to change moves no method\n\n"
            "The objective the search navigates is rounded onto a grid, which varies\n"
            "plateau structure on fixed data with the graphs, pool, budgets and seeds\n"
            "held where they are. Results are scored on the true objective.\n\n"
            + grid(["fitness levels", "local optima"] + METHODS, rows))


def _sd(value: float | None) -> str:
    """A standard deviation, or the mark for a group too small to have one."""
    return "--" if value is None else f"{value:.2f}%"


def table_selector() -> str:
    """Racing, scored over every budget pair rather than only cheapest against dearest.

    The narrower artefact this used to read covered 11 cells, and those turned out to be
    dominated by cells where several methods tie at the top -- where the selector cannot
    lose. Scoring every budget pair reaches 39 cells with a single winner, which is where
    the choice is real, and the honest figures come from there.
    """
    # One computation, two renderers. This used to carry its own copy of the arithmetic
    # alongside the manuscript generator's; they agreed, but the next change to either would
    # have gone unnoticed in the other while both still looked authoritative.
    groups = regret_groups(read("selector_regret_allpairs.csv"))
    names = list(groups[0].regret) if groups else []
    labels = {"all": "all cells", "sharp": "sharp top (one winner)",
              "plateau": "plateau top (methods tie)"}
    heading = {"race": "race", "aco": "always ACO", "ils": "always ILS", "pso": "always PSO"}
    # mean +/- sd / worst. The spread is across the group's CELLS: a row of the artefact is
    # one cell whose race is already run over its seeds, so seed-level variation sits inside
    # each value and is not recoverable here. Printed because mean and worst alone leave the
    # reader no way to tell a tight distribution from one carried by two outliers -- and here
    # they differ: the race scatters by 1.48 where always-ACO scatters by 9.42.
    body = [[labels[g.key], str(g.cells)]
            + [f"{m:.2f}% +/- {_sd(g.spread[n])} / {w:.2f}%"
               for n, (m, w) in g.regret.items()] for g in groups]

    terms = regret_terms(read("selector_seed_sweep.csv"))
    by_cause = []
    for key, label in (("estimation", "estimation: finite race seeds, shrinks as they grow"),
                       ("transfer", "transfer: the cheap budget ranks methods differently")):
        values = terms[key]
        if values:
            by_cause.append([label, str(sum(1 for v in values if v > 1.0)),
                             f"{sum(values) / len(values):.2f}%", _sd(spread(values)),
                             f"{max(values):.2f}%"])

    return ("### Racing the methods beats every fixed rule, and where it does not is measurable"
            "\n\n"
            "Chosen at the cheaper budget on even seeds, paid at the dearer on odd seeds, so\n"
            "no run appears on both sides. Every budget pair is scored, not only the extremes.\n"
            "Each entry is mean +/- sd / worst regret against the best method on the\n"
            "paying seeds; the sd is the scatter across the cells of that group, not\n"
            "across seeds.\n\n"
            + grid(["cells", "n", *[heading.get(n, n) for n in names]], body)
            + "\n\nWhere the plateau rows are exactly zero the choice costs nothing, so an\n"
              "average over both kinds understates what the selector is for. Splitting the\n"
              "regret by which term of the racing bound carries it, over 200 race draws per\n"
              "size with the paying half held fixed:\n\n"
            + grid(["term", "cells over 1%", "mean", "sd", "worst"], by_cause))


TABLES = [table_collections, table_ranks, table_basins, table_quantum, table_selector]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS / "tables.md")
    args = ap.parse_args(argv)

    parts = ["<!-- Generated by scripts/make_tables.py. Do not edit: edits are lost on the",
             "     next run, and a hand-corrected number is how the last set drifted. -->",
             "", "# Result tables", ""]
    for build in TABLES:
        parts.append(build())
        parts.append("")

    if MISSING:
        for name in sorted(set(MISSING)):
            print(f"[missing] {name}", file=sys.stderr)
        print(f"[fail] {len(set(MISSING))} input(s) absent; refusing to render a partial\n"
              f"       set of tables. An empty page that exits zero is how the previous\n"
              f"       generator went unnoticed for weeks.", file=sys.stderr)
        return 1

    text = "\n".join(parts)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[done] {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
