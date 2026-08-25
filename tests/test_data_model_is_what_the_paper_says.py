"""What the collections actually are, checked rather than described.

The manuscript called the data a set of labelled **multigraphs**, meaning parallel edges, and
the project is even named for that. The representation cannot hold them: `Graph.of` keys
edges by the vertex pair, so a second edge between the same pair silently overwrites the
first, and the loaders say they deduplicate. This holds both ends of that to measurement --
the files, and the structure -- so a claim about the data model cannot drift from the data.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from wfsmbench.core.graph import Graph

DATA = Path(__file__).resolve().parents[1] / "data"
FILES = sorted(DATA.glob("*.json"))


@pytest.mark.skipif(not FILES, reason="no collections on this machine")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_no_collection_carries_a_parallel_edge(path: Path) -> None:
    payload = json.loads(path.read_text())
    graphs = payload["graphs"] if isinstance(payload, dict) else payload
    parallel = total = 0
    for graph in graphs[:200]:
        seen: collections.Counter = collections.Counter()
        for edge in graph["edges"]:
            u, v = edge["src"], edge["dst"]
            seen[(min(u, v), max(u, v))] += 1
        total += len(graph["edges"])
        parallel += sum(count - 1 for count in seen.values() if count > 1)
    assert total > 0, f"{path.name} has no edges to check; the test would be vacuous"
    assert parallel == 0, (
        f"{path.name} carries {parallel} parallel edges out of {total}. The manuscript may "
        f"now say multigraph -- but Graph.of would drop them, so check the loader too."
    )


def test_the_representation_cannot_hold_a_parallel_edge() -> None:
    """The stronger half: even given one, the structure keeps a single edge per pair.

    If this ever fails the representation has changed and the manuscript's description of
    the data model has to be revisited, which is the point of pinning it here.
    """
    graph = Graph.of({0: 1, 1: 2}, [(0, 1, 5, 1.0), (0, 1, 7, 2.0)])
    assert graph.num_edges() == 1, (
        "two edges between one pair survived; the model is now a multigraph and section 2 "
        "of the manuscript should say so"
    )
    assert graph.edge_label(0, 1) == 7, "the later edge should win, as a dict assignment does"
