"""gSpan against an enumerator that shares none of its machinery.

The paper leaves two things unproved: that the minimum DFS code is a canonical label
(asserted, proof omitted) and that a non-minimal code's subtree holds no minimum code, which
is what licenses the pruning and is proved only in a technical report this project does not
hold. Both are therefore checked by measurement.

The check is worth something because `mine_exhaustive` uses no DFS codes at all -- it grows
patterns and deduplicates by a canonical signature computed a different way. If the pruning
dropped a subtree it should have kept, or if two non-isomorphic graphs shared a minimum code,
the two sets would differ. They do not, and the supports agree too.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.baselines.gspan import code_to_graph, gspan, min_dfs_code
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params
from wfsmbench.core.graph import Graph
from wfsmbench.core.signature import canonical_signature
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics

DATA = Path(__file__).resolve().parents[1] / "data"


def collection(name: str, limit: int) -> list[Graph]:
    path = DATA / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name} is not on this machine")
    graphs, _ = load_dataset(path)
    return graphs[:limit]


@pytest.mark.parametrize("name,limit,bound,support",
                         [("Cuneiform", 60, 3, 30), ("Cuneiform", 60, 4, 30),
                          ("QM9_bond", 200, 4, 20)])
def test_gspan_finds_exactly_what_exhaustive_enumeration_finds(
        name: str, limit: int, bound: int, support: int) -> None:
    subset = collection(name, limit)
    params = Params(min_support=support, max_nodes=bound, max_edges=bound,
                    size_reference=STUDY_SIZE_REFERENCE)
    exhaustive = {canonical_signature(c.particle.pattern): c.evaluation.support
                  for c in mine_exhaustive(subset, build_statistics(subset), params).pool}
    mined = {canonical_signature(code_to_graph(code)): value
             for code, value in gspan(subset, support, bound, bound)}

    assert len(exhaustive) >= 20, (
        f"only {len(exhaustive)} patterns at this setting; the comparison would be thin"
    )
    only_exhaustive = sorted(exhaustive.keys() - mined.keys())
    only_gspan = sorted(mined.keys() - exhaustive.keys())
    assert not only_exhaustive, (
        f"gSpan missed {len(only_exhaustive)} patterns the enumerator found, so the "
        f"non-minimal-code pruning is dropping subtrees it should keep: {only_exhaustive[:3]}"
    )
    assert not only_gspan, (
        f"gSpan reported {len(only_gspan)} patterns the enumerator did not: {only_gspan[:3]}"
    )
    disagreeing = [(k, exhaustive[k], mined[k]) for k in exhaustive if exhaustive[k] != mined[k]]
    assert not disagreeing, f"same patterns, different support: {disagreeing[:3]}"


def test_the_minimum_code_is_invariant_under_renaming_vertices() -> None:
    """Canonicity of the minimum DFS code, in the form that can fail.

    Comparing minimum codes across the mined pool alone proves nothing: that pool is already
    deduplicated by canonical signature, so no two of its patterns are isomorphic and the
    check cannot fail. Renaming the vertices makes isomorphic pairs on purpose.
    """
    subset = collection("Cuneiform", 60)
    params = Params(min_support=30, max_nodes=4, max_edges=4,
                    size_reference=STUDY_SIZE_REFERENCE)
    pool = mine_exhaustive(subset, build_statistics(subset), params).pool
    rng = random.Random(0)
    checked = 0
    for candidate in pool[:120]:
        pattern = candidate.particle.pattern
        expected = min_dfs_code(pattern)
        for _ in range(4):
            ids = list(pattern.node_labels)
            shuffled = ids[:]
            rng.shuffle(shuffled)
            rename = dict(zip(ids, shuffled))
            renamed = Graph.of(
                {rename[n]: lab for n, lab in pattern.node_labels.items()},
                [(rename[u], rename[v], lab, pattern.weight(u, v))
                 for (u, v), lab in pattern.edge_items()])
            checked += 1
            assert min_dfs_code(renamed) == expected, (
                f"renaming the vertices changed the minimum code, so it is not canonical"
            )
    assert checked >= 200, f"only {checked} renamings; the check would be thin"


def test_distinct_patterns_get_distinct_codes() -> None:
    """The other half of canonicity: non-isomorphic graphs must not collide."""
    subset = collection("Cuneiform", 60)
    params = Params(min_support=30, max_nodes=4, max_edges=4,
                    size_reference=STUDY_SIZE_REFERENCE)
    pool = mine_exhaustive(subset, build_statistics(subset), params).pool
    # An empty pool makes 0 == 0 == 0 below, which is a pass over nothing.
    assert len(pool) >= 20, (
        f"only {len(pool)} pattern(s) mined; the collision check needs a pool to collide in")
    codes = {min_dfs_code(c.particle.pattern) for c in pool}
    signatures = {canonical_signature(c.particle.pattern) for c in pool}
    assert len(codes) == len(signatures) == len(pool), (
        f"{len(pool)} patterns collapse to {len(codes)} codes and {len(signatures)} "
        f"signatures; the two canonical forms disagree about which graphs are the same"
    )


def test_the_pruning_removes_most_of_the_code_space() -> None:
    """That the prune is implemented, not merely present.

    Removing the prune cannot lose a pattern -- it only lets each graph be reached under
    several codes -- so the agreement test above stays green without it. That test catches
    the direction that matters, over-pruning, and this one catches the other: a prune that
    fires on nothing would leave the algorithm correct and pointless.
    """
    import wfsmbench.baselines.gspan as module

    subset = collection("Cuneiform", 60)
    pruned = module.gspan(subset, 30, 4, 4)

    class AlwaysMatches:
        def __eq__(self, other: object) -> bool:
            return True

    real = module.min_dfs_code
    module.min_dfs_code = lambda _graph: AlwaysMatches()
    try:
        unpruned = module.gspan(subset, 30, 4, 4)
    finally:
        module.min_dfs_code = real

    assert len(unpruned) > len(pruned), "the prune fires on nothing"
    removed = (len(unpruned) - len(pruned)) / len(unpruned)
    assert removed > 0.5, (
        f"the prune removes only {removed:.0%} of the codes reached; at that rate it is not "
        f"doing the work the paper credits it with"
    )
    print(f"\n[prune] {len(unpruned)} codes without it, {len(pruned)} with, "
          f"{removed:.0%} removed")
