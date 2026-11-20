"""The peak label follows the manuscript's definition: at the bound is the pool's largest edge count."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_degeneracy as rd  # noqa: E402


def test_largest_edge_count_of_the_pool_is_at_the_bound() -> None:
    # K = 4 but the pool stops at 3 edges: the 3-edge peak cannot grow, so it is at the bound
    assert rd.shape_of(3, 3) == "at-the-bound"
    assert rd.shape_of(4, 4) == "at-the-bound"


def test_below_the_pool_maximum_is_interior_and_one_edge_is_the_floor() -> None:
    assert rd.shape_of(3, 4) == "interior"
    assert rd.shape_of(1, 4) == "at-the-floor"
