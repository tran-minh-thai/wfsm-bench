r"""Build weighted, sparse molecular graphs from TUDataset's raw molecular collections.

TUDataset ships two shapes of molecular data and neither is what this problem needs on its
own. The ``_MD`` collections (BZR_MD, COX2_MD, DHFR_MD, ER_MD) carry edge weights but join
*every* pair of atoms by distance, so the graphs are complete -- E/N of 10.6 to 12.8 -- and on
a complete graph every combination of labels occurs in every graph, support is near-constant,
and the WSUP term loses its power to discriminate. The plain collections (BZR, COX2, DHFR)
carry the real bond graph, which is sparse, but no edge weights at all.

The weights are there, though, in a column nobody reads as one: ``_node_attributes.txt`` holds
3D atomic coordinates, so the length of a bond is the distance between the two atoms it joins.
Taking bonds only -- not all pairs -- and weighting each by that distance gives what the model
asks for. Measured: E/N falls to 1.05-1.07, and the bond lengths land where chemistry says
they should, with medians of 1.35 to 1.39 A against C-C 1.54, C=C 1.34, C-N 1.47.

**This is a design decision that can be wrong, not a harmless preprocessing step.** Each raw
column is assigned a role, and a different assignment gives a different problem:

    _graph_indicator.txt        which molecule -- one transaction
    _node_labels.txt or an      node label
      atom-number column
    _A.txt                      edges: the real bonds, never all pairs
    _edge_attributes.txt        edge label, where the raw data has bond types
    coordinate columns of       edge weight: Euclidean distance between bonded atoms
      _node_attributes.txt

The role that hurts when it is missing is the edge label. BZR, COX2 and DHFR have none, so a
constant is assigned, templates collapse to bare atom-type pairs, and the pattern space
thins out: 14 to 16 frequent templates against Cuneiform's 24, and a pool of 62 patterns at
four edges against 712. Those three are usable as degenerate-case controls, not as evidence
for the model. QM9 carries bond types in its edge attributes, so its edge label is read
rather than invented.

An edge label must never be derived from the bond length. The length is already the weight,
and deriving the label from it would let the weight gate matching through the back door,
which the design forbids -- weights score, they do not decide what matches what.

The dataset table in README.md lists the collections built this way; the measurements
behind it are in results-invariant/dataset_profile.csv.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..core.graph import Graph
from .tudataset import WEIGHT_MAX, WEIGHT_MIN


@dataclass(frozen=True)
class Layout:
    """Which raw column plays which role, for one collection.

    ``coordinates`` indexes into ``_node_attributes.txt``. ``node_label_column`` selects a
    node-attribute column to label by; None means read ``_node_labels.txt`` instead.
    ``edge_label_from_attributes`` reads the bond type as the argmax of the one-hot in
    ``_edge_attributes.txt``; False assigns a constant, which is what the collections without
    bond types have to do.
    """

    coordinates: tuple[int, int, int]
    node_label_column: int | None = None
    edge_label_from_attributes: bool = False
    # Some collections publish the bond type directly as an integer per edge instead of
    # as a one-hot attribute row. Same role, different file, so it needs its own flag
    # rather than a guess at which file happens to exist.
    edge_label_from_labels: bool = False


# The role assignments actually used, recorded here rather than in a script so they are
# reviewable and reproducible. Column indices are 0-based into the raw attribute rows.
LAYOUTS: dict[str, Layout] = {
    # 3 columns, all of them the position.
    "BZR": Layout(coordinates=(0, 1, 2)),
    "COX2": Layout(coordinates=(0, 1, 2)),
    "DHFR": Layout(coordinates=(0, 1, 2)),
    # 16 columns: one-hot atom type, then the atomic number, then more chemistry, then the
    # position last. The atomic number is the cleaner label -- one column, no decoding.
    "QM9": Layout(coordinates=(13, 14, 15), node_label_column=5,
                  edge_label_from_attributes=True),
    # 3 columns, all position; atom type in _node_labels.txt and bond type in
    # _edge_labels.txt, both as plain integers. This is the only collection carrying
    # published bond types *and* geometry without needing either decoded from a one-hot.
    "alchemy_full": Layout(coordinates=(0, 1, 2), edge_label_from_labels=True),
}


def _read(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_graphs(graph_of_node: list[int],
                 node_labels: list[int],
                 positions: list[tuple[float, ...]],
                 edge_pairs: list[tuple[int, int]],
                 edge_labels: list[int] | None = None,
                 limit: int | None = None) -> list[dict]:
    """Assemble the per-graph structure from already-parsed raw rows.

    Split out from the file reading so it can be tested without a downloaded dataset. Node
    ids are 1-based, as TUDataset writes them. Edges are undirected and deduplicated: the
    raw file lists both directions, and a bond is one edge.
    """
    per_graph: dict[int, list[int]] = defaultdict(list)
    for global_id, graph_id in enumerate(graph_of_node, start=1):
        if limit is None or graph_id <= limit:
            per_graph[graph_id].append(global_id)
    local_of = {global_id: (graph_id, index)
                for graph_id, ids in per_graph.items()
                for index, global_id in enumerate(ids)}

    edges: dict[int, dict[tuple[int, int], tuple[int, float]]] = defaultdict(dict)
    for line, (u, v) in enumerate(edge_pairs):
        if u not in local_of or v not in local_of:
            continue
        graph_u, local_u = local_of[u]
        graph_v, local_v = local_of[v]
        if graph_u != graph_v or local_u == local_v:
            continue
        key = (local_u, local_v) if local_u < local_v else (local_v, local_u)
        if key in edges[graph_u]:
            continue
        label = 0 if edge_labels is None else edge_labels[line]
        edges[graph_u][key] = (label, math.dist(positions[u - 1], positions[v - 1]))

    lengths = [length for graph in per_graph for _label, length in edges[graph].values()]
    if not lengths:
        raise ValueError("no bonds found; check the role assignment")
    low, high = min(lengths), max(lengths)
    span = high - low

    def weight(length: float) -> float:
        if span == 0.0:
            return WEIGHT_MIN
        return WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * (length - low) / span

    seen_labels = sorted({node_labels[global_id - 1]
                          for ids in per_graph.values() for global_id in ids})
    label_index = {value: index for index, value in enumerate(seen_labels)}

    built = []
    for new_id, graph_id in enumerate(sorted(per_graph)):
        ids = per_graph[graph_id]
        built.append({
            "id": new_id,
            "vertices": [{"id": index, "label": label_index[node_labels[global_id - 1]]}
                         for index, global_id in enumerate(ids)],
            "edges": [{"src": a, "dst": b, "label": label, "weight": round(weight(length), 6)}
                      for (a, b), (label, length) in sorted(edges[graph_id].items())],
        })
    return built, seen_labels


def convert(raw_dir: str | Path, name: str, limit: int | None = None) -> dict:
    """Read one raw collection and return the project's dataset JSON."""
    if name not in LAYOUTS:
        raise ValueError(f"no role assignment for {name!r}; known: {sorted(LAYOUTS)}")
    layout = LAYOUTS[name]
    base = Path(raw_dir) / name

    graph_of_node = [int(x) for x in _read(base / f"{name}_graph_indicator.txt")]
    attributes = [[float(v) for v in line.split(",")]
                  for line in _read(base / f"{name}_node_attributes.txt")]
    positions = [tuple(row[i] for i in layout.coordinates) for row in attributes]
    if layout.node_label_column is None:
        node_labels = [int(x) for x in _read(base / f"{name}_node_labels.txt")]
    else:
        node_labels = [int(row[layout.node_label_column]) for row in attributes]
    edge_pairs = [tuple(int(x) for x in line.split(","))
                  for line in _read(base / f"{name}_A.txt")]
    edge_labels = None
    if layout.edge_label_from_labels:
        edge_labels = [int(float(line.split(",")[0]))
                       for line in _read(base / f"{name}_edge_labels.txt")]
    elif layout.edge_label_from_attributes:
        rows = [[float(v) for v in line.split(",")]
                for line in _read(base / f"{name}_edge_attributes.txt")]
        edge_labels = [max(range(len(row)), key=row.__getitem__) for row in rows]

    graphs, seen_labels = build_graphs(graph_of_node, node_labels, positions,
                                       edge_pairs, edge_labels, limit)
    return {
        "name": f"{name.lower()}_bond",
        "description": (
            f"{name} from TUDataset, rebuilt as a weighted sparse molecular graph: edges are "
            f"the real bonds only, and each weight is the Euclidean distance between the two "
            f"bonded atoms, min-max scaled to [{WEIGHT_MIN}, {WEIGHT_MAX}]. Edge labels are "
            + ("the raw bond types." if layout.edge_label_from_attributes
               else "constant -- the raw data carries none.")
            + " See io/molecular.py for the role assignment."),
        "label_strings": [str(value) for value in seen_labels],
        "graphs": graphs,
    }


def convert_and_save(raw_dir: str | Path, name: str, out_path: str | Path,
                     limit: int | None = None) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert(raw_dir, name, limit), indent=2), encoding="utf-8",
                   newline="\n")
    return out


def load_graphs(raw_dir: str | Path, name: str, limit: int | None = None) -> list[Graph]:
    """Convert and return graphs directly, without writing JSON."""
    data = convert(raw_dir, name, limit)
    return [Graph.of({v["id"]: v["label"] for v in gj["vertices"]},
                     [(e["src"], e["dst"], e["label"], e["weight"]) for e in gj["edges"]],
                     graph_id=gj["id"])
            for gj in data["graphs"]]
