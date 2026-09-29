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


PROFILE = (Path(__file__).resolve().parents[1] / "results-invariant" / "dataset_profile.csv")


def _scan(path: Path) -> tuple[int, int, int]:
    """(graphs, edges, parallel edges) over the WHOLE file."""
    payload = json.loads(path.read_text())
    graphs = payload["graphs"] if isinstance(payload, dict) else payload
    parallel = total = 0
    for graph in graphs:
        seen: collections.Counter = collections.Counter()
        for edge in graph["edges"]:
            u, v = edge["src"], edge["dst"]
            seen[(min(u, v), max(u, v))] += 1
        total += len(graph["edges"])
        parallel += sum(count - 1 for count in seen.values() if count > 1)
    return len(graphs), total, parallel


@pytest.mark.skipif(not FILES, reason="no collections on this machine")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_no_collection_carries_a_parallel_edge(path: Path) -> None:
    """CORRECTED 2026-09-02: scan the whole file, not its first two hundred graphs.

    The previous version stopped at graphs[:200] while the manuscript states the property of
    the whole data. Every collection here is larger than that -- 306 to 5000 graphs -- so the
    scan covered 1600 of 12674 graphs, twelve per cent, and claimed all of it. Removing the
    cap costs 0.3 seconds for the entire suite of eight files, so the cap bought nothing.
    """
    graphs, total, parallel = _scan(path)
    assert total > 0, f"{path.name} has no edges to check; the test would be vacuous"
    assert parallel == 0, (
        f"{path.name} carries {parallel} parallel edges out of {total} across {graphs} "
        f"graphs. The manuscript may now say multigraph -- but Graph.of would drop them, so "
        f"check the loader too."
    )


@pytest.mark.skipif(not (FILES and PROFILE.exists()),
                    reason="no collections or no profile on this machine")
def test_the_scan_here_agrees_with_the_artefact_the_paper_quotes() -> None:
    """Tie the check to the number in the manuscript, so the two cannot drift apart.

    The paper quotes a total raw edge count and a parallel-edge count from
    results-invariant/dataset_profile.csv. If this test and that artefact disagree, one of
    them is measuring something else, and reading either alone would not say which.
    """
    import csv
    rows = list(csv.DictReader(PROFILE.open(encoding="utf-8")))
    assert rows, "the dataset profile is empty; this check would compare nothing"
    artefact_profile = {r["dataset"]: (int(r["graphs"]), int(r["edges_total_raw"]),
                               int(r["parallel_edges"])) for r in rows}

    # CORRECTED 2026-09-08: compare COLLECTION BY COLLECTION, not by one grand total.
    #
    # The total was wrong in two ways at once. It could not say WHICH collection had drifted
    # -- the message gave two triples and left the reader to subtract -- and it could be
    # right while both sides were wrong, because a collection reading too high cancels one
    # reading too low. Measured on the experiment machine: the totals differed by exactly
    # (473, 76667, 0), which is one whole collection absent, and the message named neither
    # the collection nor the fact that anything was missing.
    disagreements = []
    for path in FILES:
        if path.stem not in artefact_profile:
            continue
        scanned = _scan(path)
        if scanned != artefact_profile[path.stem]:
            disagreements.append((path.stem, scanned, artefact_profile[path.stem]))
    assert not disagreements, (
        f"{len(disagreements)} of {len(FILES)} collection(s) on this machine disagree with the "
        f"artefact the paper quotes. One of the two is measuring something else:\n"
        + "\n".join(f"  {collection_name}: this scan reads {a} (graphs, edges, parallel), "
                     f"the artefact says {b}" for collection_name, a, b in disagreements))

    present_here = {path.stem for path in FILES}
    absent_here = sorted(set(artefact_profile) - present_here)
    assert not absent_here, (
        f"{len(absent_here)} of {len(artefact_profile)} collection(s) the paper quotes are NOT on this "
        f"machine: {absent_here}. Every collection present agrees, so nothing has drifted -- the "
        f"data is simply absent, and any stage needing it cannot run here. "
        f"scripts/prepare_datasets.py builds the TUDataset collections; anything it does "
        f"not build is tracked in data/ and arrives with a pull.")

    unknown_here = sorted(present_here - set(artefact_profile))
    assert not unknown_here, (
        f"{len(unknown_here)} collection(s) on this machine are absent from the profile the paper "
        f"quotes: {unknown_here}. Either the profile is stale or a collection was added without "
        f"being profiled; a number quoted from a stale profile describes data nobody has.")


def test_the_representation_cannot_hold_a_parallel_edge() -> None:
    """The stronger half: even given one, the structure keeps a single edge per pair.

    If this ever fails the representation has changed and the manuscript's description of
    the data model has to be revisited, which is the point of pinning it here.
    """
    graph = Graph.of({0: 1, 1: 2}, [(0, 1, 5, 1.0), (0, 1, 7, 2.0)])
    assert graph.num_edges() == 1, (
        "two edges between one pair survived; the model is now a multigraph and its "
        "written description should say so"
    )
    assert graph.edge_label(0, 1) == 7, "the later edge should win, as a dict assignment does"
