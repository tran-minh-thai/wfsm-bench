"""One enumeration can stand in for many, and this holds it to that.

The Pareto-selection sweep asks the same question at 21 (size bound, support) cells. Mining
each cell separately re-enumerates overlapping search spaces 21 times, and on the largest
cell that is 42,523 patterns. Three properties of the existing code say one loose enumeration
covers them all:

  * ``is_valid`` rejects on ``num_edges() > max_edges`` and ``num_nodes() > max_nodes``, so a
    pattern admitted under a tight bound is admitted under a looser one;
  * ``frequent_edge_templates(min_support)`` returns more templates as the threshold falls,
    and ``mine_exhaustive`` only expands patterns that clear the threshold, so a low
    threshold explores a superset;
  * ``fitness`` anchors on ``Params.size_reference``, a constant of the study, and never on
    ``max_edges`` -- so a pattern's F does not depend on the bound it was found under.

Those are three readings of the source, which is an argument. This is the measurement: mine
the loose cell, filter it, and require the result to equal a direct mining of the tight cell
exactly -- same patterns, same support, same fitness.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params
from wfsmbench.core.signature import canonical_signature
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics

DATA = Path(__file__).resolve().parents[1] / "data" / "Cuneiform.json"
LIMIT = 30          # measured: bound 3 here mines in ~3.5 s, bound 4 in ~53 s
LOOSE_BOUND, LOOSE_SUPPORT = 3, 7


def params_for(bound: int, min_support: int) -> Params:
    """Both bounds equal the size, matching run_weight_mode.py and hence the 21 cells."""
    return Params(min_support=min_support, max_nodes=bound, max_edges=bound,
                  size_reference=STUDY_SIZE_REFERENCE)


def keyed(pool) -> dict[str, tuple[int, float]]:
    return {canonical_signature(c.particle.pattern):
            (c.evaluation.support, round(c.fitness, 12)) for c in pool}


def derived(loose_pool, bound: int, min_support: int) -> dict[str, tuple[int, float]]:
    return {sig: value for sig, value in keyed(loose_pool).items() if value[0] >= min_support}


@pytest.fixture(scope="module")
def collection():
    graphs, _ = load_dataset(DATA)
    subset = graphs[:LIMIT]
    return subset, build_statistics(subset)


@pytest.fixture(scope="module")
def loose_pool(collection):
    """Mined once: the whole point is that one enumeration serves every cell."""
    subset, stats = collection
    return mine_exhaustive(subset, stats, params_for(LOOSE_BOUND, LOOSE_SUPPORT)).pool


@pytest.mark.parametrize("tight_bound,tight_support", [(2, 7), (2, 15), (3, 15), (3, 21)])
def test_a_loose_enumeration_reproduces_a_tight_one_exactly(
        collection, loose_pool, tight_bound, tight_support):
    subset, stats = collection
    loose = loose_pool
    direct = mine_exhaustive(subset, stats, params_for(tight_bound, tight_support)).pool

    kept = {sig: value for sig, value in keyed(loose).items() if value[0] >= tight_support}
    # The size filter has to be applied on the pattern, not on the key, so redo it properly.
    sizes = {canonical_signature(c.particle.pattern):
             (c.particle.pattern.num_edges(), c.particle.pattern.num_nodes())
             for c in loose}
    kept = {sig: value for sig, value in kept.items()
            if sizes[sig][0] <= tight_bound and sizes[sig][1] <= tight_bound}

    expected = keyed(direct)
    assert kept.keys() == expected.keys(), (
        f"bound={tight_bound} support={tight_support}: derived {len(kept)} patterns, "
        f"direct mining gives {len(expected)}; "
        f"only-derived={sorted(kept.keys() - expected.keys())[:3]} "
        f"only-direct={sorted(expected.keys() - kept.keys())[:3]}"
    )
    assert kept == expected, "same patterns, but support or fitness differs"
    assert expected, "the tight cell produced an empty pool -- this check would be vacuous"


def test_the_filter_is_not_trivially_everything(collection, loose_pool):
    """If the loose pool equalled the tight one the test above would prove nothing."""
    subset, stats = collection
    loose = loose_pool
    direct = mine_exhaustive(subset, stats, params_for(2, 15)).pool
    assert len(loose) > len(direct) * 2, (
        f"loose pool {len(loose)} is not meaningfully larger than tight {len(direct)}; "
        "pick a wider gap or the nesting check is vacuous"
    )
