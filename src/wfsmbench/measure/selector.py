"""What the racing selector costs, computed once for every renderer that reports it.

Two scripts report these figures -- one builds the result tables, the other fills the
manuscript's generated blocks -- and each had grown its own copy of the arithmetic. They
agreed when checked, but two implementations of one measurement is a defect waiting for the
first change to only one of them: regroup the cells in one place and the other keeps the old
grouping while still looking authoritative.

So the computation lives here and the renderers only format. Reading the files stays with the
callers, since they resolve paths differently.

FIXED RULES ARE ALL REPORTED, and this is not a presentation choice. Racing is worth its cost
only against the best fixed rule available, not against a convenient one: always-ACO loses
5.13% on average where always-ILS loses 3.11%, so quoting only the first would overstate the
selector by nearly a factor of two. Any strategy column present in the artefact is returned.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# Cell groups, keyed rather than labelled: the label is the renderer's business and differs
# by language, while the predicate is the measurement and must not.
GROUPS = {
    "all": lambda row: True,
    "sharp": lambda row: row["top"] == "sharp",
    "plateau": lambda row: row["top"] == "plateau",
}

# The column holding the selector itself, and the prefix marking a fixed-rule comparison.
RACE_COLUMN = "regret_race"
FIXED_PREFIX = "regret_"

# Race size at which the bound's two terms are separated. Ten is where the estimation term
# has effectively run out on every cell that has the seeds for it.
DECOMPOSITION_SIZE = 10


@dataclass(frozen=True)
class Group:
    key: str
    cells: int
    # strategy key -> (mean regret, worst regret), in percent
    regret: dict[str, tuple[float, float]]
    # strategy key -> sample standard deviation of the same values, in percent. None where a
    # group holds fewer than two cells, since one cell has no dispersion to report and a
    # printed 0.00 there would read as "no variation" rather than "nothing to vary".
    spread: dict[str, float | None]


def spread(values: list[float]) -> float | None:
    """Sample standard deviation, or None when there is nothing to spread.

    Deliberately NOT a confidence interval. These regrets are bounded below by zero, most
    cells sit exactly at zero, and the tail is long -- the plateau group is 22 cells of
    which every one is 0.00 for the race. A t interval over that shape asserts a symmetric
    distribution the data does not have, and would print a negative lower bound for a
    quantity that cannot be negative. The standard deviation makes no such claim: it says
    how far the cells scatter, and the worst column already gives the tail.
    """
    return statistics.stdev(values) if len(values) > 1 else None


def strategies(rows: list[dict]) -> list[str]:
    """Every strategy the artefact scores, race first and the fixed rules after it."""
    if not rows:
        return []
    fixed = sorted(name[len(FIXED_PREFIX):] for name in rows[0]
                   if name.startswith(FIXED_PREFIX) and name != RACE_COLUMN)
    return ["race", *fixed]


def regret_groups(rows: list[dict]) -> list[Group]:
    """Mean and worst regret per cell group, for the race and for every fixed rule."""
    disjoint = [r for r in rows if r["seed_split"] == "disjoint"]
    names = strategies(disjoint)
    out = []
    for key, keep in GROUPS.items():
        kept = [r for r in disjoint if keep(r)]
        if not kept:
            continue
        regret, spreads = {}, {}
        for name in names:
            column = RACE_COLUMN if name == "race" else f"{FIXED_PREFIX}{name}"
            values = [float(r[column]) for r in kept if r.get(column)]
            if values:
                regret[name] = (sum(values) / len(values), max(values))
                # Over CELLS, not over seeds. A row of the artefact is one cell whose race
                # has already been run over its seeds, so the seed-level variation is inside
                # each value and cannot be recovered from here. What scatters across the
                # rows is the cell, and that is what the column has to be read as.
                spreads[name] = spread(values)
        out.append(Group(key=key, cells=len(kept), regret=regret, spread=spreads))
    return out


def regret_terms(rows: list[dict]) -> dict[str, list[float]]:
    """The two terms of the racing bound, separated per cell.

    The difference is taken inside each cell before anything is aggregated. Taking the worst
    regret and the worst floor separately and subtracting would subtract across two different
    cells.
    """
    estimation, transfer = [], []
    column = f"regret_n{DECOMPOSITION_SIZE}"
    for row in rows:
        if not (row.get(column) and row.get("transfer_only")):
            continue
        floor = float(row["transfer_only"])
        transfer.append(floor)
        estimation.append(float(row[column]) - floor)
    return {"estimation": estimation, "transfer": transfer}
