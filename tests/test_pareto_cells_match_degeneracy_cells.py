"""The Pareto sweep must ask about the same 21 cells the degeneracy result reports on.

Two settings decide which cells those are, and both were wrong on the first attempt:

  * ``run_weight_mode.py`` builds ``Params(max_nodes=size, max_edges=size)`` -- **both**
    bounds equal the size. Allowing one extra node quadrupled the pool at bound 4, 2959
    against 712.
  * every ``min_support`` in ``results/weight_mode_landscape.csv`` is a fraction of **60**,
    and its ``graph_limit`` column says 60. The whole collection is 267, so running it
    answers a different question under the same cell names, at about four times the cost.

Neither error changes a column name or raises anything. Both were caught only by comparing
pool sizes against the artefact, which is what this does.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANDSCAPE = ROOT / "results" / "weight_mode_landscape.csv"

import sys
sys.path.insert(0, str(ROOT / "scripts"))


def landscape_rows() -> list[dict]:
    with LANDSCAPE.open(newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["weight_mode"] == "sum"]


@pytest.mark.skipif(not LANDSCAPE.exists(), reason=f"artefact not present: {LANDSCAPE}")
def test_the_sweep_defaults_to_the_graph_limit_the_cells_were_run_on() -> None:
    import run_pareto_size

    limits = {int(r["graph_limit"]) for r in landscape_rows()}
    assert limits == {run_pareto_size.STUDY_GRAPH_LIMIT}, (
        f"the degeneracy cells were run at graph_limit {sorted(limits)}, but the sweep "
        f"defaults to {run_pareto_size.STUDY_GRAPH_LIMIT}"
    )


@pytest.mark.skipif(not LANDSCAPE.exists(), reason=f"artefact not present: {LANDSCAPE}")
def test_the_support_fractions_reproduce_the_recorded_thresholds() -> None:
    import run_pareto_size

    limit = run_pareto_size.STUDY_GRAPH_LIMIT
    derived = {max(1, int(round(f * limit))) for f in run_pareto_size.SUPPORT_FRACTIONS}
    recorded = {int(r["min_support"]) for r in landscape_rows()}
    assert derived == recorded, (
        f"the sweep would mine at min_support {sorted(derived)}; the cells were run at "
        f"{sorted(recorded)}"
    )


@pytest.mark.skipif(not LANDSCAPE.exists(), reason=f"artefact not present: {LANDSCAPE}")
def test_the_size_bounds_match() -> None:
    recorded = sorted({int(r["size_bound"]) for r in landscape_rows()})
    assert recorded == [4, 5, 6], f"artefact bounds {recorded} are not the documented 4, 5, 6"
    assert len(landscape_rows()) == 21, "the artefact no longer holds 21 cells"
