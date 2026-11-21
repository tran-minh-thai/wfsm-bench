"""The designs of the supplementary selector runs (docs/SUPPLEMENTARY_PROTOCOL.md, T1 and T2).

Checked on the selector table's own cells: successive halving never spends more than the race
it is compared with and starts from the largest budget that fits; the restart count is the
selector's cost divided by one paid run; the fresh seeds collide with no racing or paying seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

R = pytest.importorskip("run_selector_supplement")


@pytest.fixture(scope="module")
def cells():
    try:
        configs = R.configurations(R.recorded_rows())
    except SystemExit:
        pytest.skip("the selector table's run files are not present")
    return R.cells(configs)


def _cost(cell, b0):
    keep, n = [7, 4, 2], len(cell["race"])
    rounds = [min(b0 * 2 ** j, cell["dear"]) for j in range(3)]
    return n * sum(m * b for m, b in zip(keep, rounds)) + n * cell["dear"]


def test_halving_fits_the_race_budget_and_is_the_largest_that_does(cells) -> None:
    assert len(cells) == 61
    for cell in cells:
        b0, _rounds, cost = R.halving_schedule(cell)
        t_race = R.M * len(cell["race"]) * cell["cheap"] + len(cell["pay"]) * cell["dear"]
        assert cost == _cost(cell, b0) <= t_race
        assert _cost(cell, b0 + 1) > t_race


def test_restarts_equal_the_selector_cost_over_one_paid_run(cells) -> None:
    for cell in cells:
        t_race = R.M * len(cell["race"]) * cell["cheap"] + len(cell["pay"]) * cell["dear"]
        assert R.restarts(cell) == t_race // (len(cell["pay"]) * cell["dear"])
        assert R.b_equal(cell) * len(cell["pay"]) == t_race


def test_fresh_seeds_collide_with_nothing(cells) -> None:
    used = {s for c in cells for s in c["race"] + c["pay"]}
    fresh = {R.extra_seed(s, j) for c in cells for s in c["pay"] for j in range(1, R.restarts(c))}
    assert fresh and not fresh & used
    assert len(fresh) == len({(s, j) for c in cells for s in c["pay"] for j in range(1, R.restarts(c))})
