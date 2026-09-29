"""The size-degeneracy counts the study quotes must be recomputed from the artefact.

The manuscript said the optimum sat at the size bound in every one of 21 configurations.
The artefact says 16. The earlier written account had it right all along -- interior optima appear
only from the 50% threshold -- so the error was in the prose, and prose is exactly what
nothing checks. This recomputes every such number from the file that produced
them, and prints its denominators.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

ARTEFACT = Path(__file__).resolve().parents[1] / "results" / "weight_mode_landscape.csv"

# The thresholds a practitioner would actually choose, as a fraction of the database.
PRACTICAL_MAX_SUPPORT = 0.30


def load(mode: str) -> list[dict]:
    with ARTEFACT.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["weight_mode"] == mode]
    assert rows, f"no rows for weight_mode={mode!r}; the artefact changed shape"
    return rows


def at_bound(rows: list[dict]) -> int:
    return sum(1 for r in rows if int(r["optimum_edges"]) == int(r["size_bound"]))


@pytest.mark.skipif(not ARTEFACT.exists(), reason=f"artefact not present: {ARTEFACT}")
def test_penalised_objective_pins_to_the_bound_in_16_of_21() -> None:
    rows = load("sum")
    assert len(rows) == 21, f"expected 21 configurations, artefact has {len(rows)}"
    assert at_bound(rows) == 16, (
        f"the study quotes 16 of 21 at the size bound; artefact gives {at_bound(rows)} of "
        f"{len(rows)}. Fix the prose, not this test."
    )


@pytest.mark.skipif(not ARTEFACT.exists(), reason=f"artefact not present: {ARTEFACT}")
def test_at_practical_thresholds_it_is_unanimous() -> None:
    rows = [r for r in load("sum") if float(r["support_fraction"]) <= PRACTICAL_MAX_SUPPORT]
    assert len(rows) == 12, f"expected 12 cells at or below 30% support, got {len(rows)}"
    assert at_bound(rows) == len(rows), (
        f"the study calls this unanimous; {at_bound(rows)} of {len(rows)} sit at the bound"
    )


@pytest.mark.skipif(not ARTEFACT.exists(), reason=f"artefact not present: {ARTEFACT}")
def test_interior_optima_appear_only_above_the_practical_range() -> None:
    interior = [r for r in load("sum") if int(r["optimum_edges"]) != int(r["size_bound"])]
    assert len(interior) == 5, f"expected 5 interior cells, got {len(interior)}"
    below = [r for r in interior if float(r["support_fraction"]) <= PRACTICAL_MAX_SUPPORT]
    assert not below, (
        "an interior optimum appeared at a practical threshold, which contradicts the study: "
        + str([(r["size_bound"], r["support_fraction"], r["optimum_edges"]) for r in below])
    )


@pytest.mark.skipif(not ARTEFACT.exists(), reason=f"artefact not present: {ARTEFACT}")
def test_dividing_by_size_inverts_rather_than_removes() -> None:
    """per_edge is weighted support over the pattern's edge count -- Yang et al.'s repair."""
    rows = load("per_edge")
    assert len(rows) == 21, f"expected 21 configurations, artefact has {len(rows)}"
    assert at_bound(rows) == 0, f"{at_bound(rows)} of {len(rows)} still sit at the bound"
    sizes = {int(r["optimum_edges"]) for r in rows}
    assert sizes == {1}, (
        f"the study says a single edge in all 21; artefact shows optimum sizes {sorted(sizes)}"
    )
