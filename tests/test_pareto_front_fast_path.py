"""The fast front must return exactly what the brute-force definition returns.

pareto_front compares each candidate only against the already-accepted front. That is a
large speedup and it is only safe because of two properties -- descending lexicographic
order, and transitivity of dominance -- which are an argument, not a measurement. This holds
it to the definition on random inputs, including the shapes where a sloppy sweep breaks:
ties on one objective, duplicate points, and fronts that are the whole input.
"""
from __future__ import annotations

import random

import pytest

from wfsmbench.measure.pareto import dominates, pareto_front


def naive_front(points: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """The definition, spelled out: keep what nothing else dominates."""
    unique = sorted(set(points), reverse=True)
    return [p for p in unique if not any(dominates(q, p) for q in unique if q != p)]


@pytest.mark.parametrize("seed", range(40))
def test_pareto_front_matches_naive(seed: int) -> None:
    rng = random.Random(seed)
    # A small value range on purpose: it forces ties and duplicates, which is where an
    # order-dependent sweep goes wrong and a wide random range would never exercise.
    points = [tuple(float(rng.randint(0, 4)) for _ in range(3))
              for _ in range(rng.randint(1, 60))]
    assert pareto_front(points) == naive_front(points), f"seed {seed}"


def test_handles_the_degenerate_shapes() -> None:
    assert pareto_front([]) == []
    assert pareto_front([(1.0, 1.0, 1.0)]) == [(1.0, 1.0, 1.0)]
    # Every point identical: one survivor, duplicates removed.
    assert pareto_front([(2.0, 2.0, 2.0)] * 5) == [(2.0, 2.0, 2.0)]
    # A total order: only the top survives.
    chain = [(float(i), float(i), float(i)) for i in range(6)]
    assert pareto_front(chain) == [(5.0, 5.0, 5.0)]
    # Mutually non-dominated: all survive, and the naive path agrees.
    spread = [(3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)]
    assert pareto_front(spread) == naive_front(spread)
    assert len(pareto_front(spread)) == 3
