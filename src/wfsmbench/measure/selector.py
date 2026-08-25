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
        regret = {}
        for name in names:
            column = RACE_COLUMN if name == "race" else f"{FIXED_PREFIX}{name}"
            values = [float(r[column]) for r in kept if r.get(column)]
            if values:
                regret[name] = (sum(values) / len(values), max(values))
        out.append(Group(key=key, cells=len(kept), regret=regret))
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
