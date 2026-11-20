"""Convert a raw TUDataset (chrsmrrs.github.io) into the project's dataset JSON.

Handles the labeled, edge-weighted datasets used in the study (BZR_MD, Cuneiform).
Edge weights follow decision D7 (docs/DesignDecisions.md): each edge's raw "distance"
is the ``*_edge_attributes.txt`` value (1-D) or the L2 norm of the attribute vector
(multi-D), then min-max scaled per dataset to ``[WEIGHT_MIN, WEIGHT_MAX]``. For BZR_MD
this reproduces the dataset statistics of the authors' earlier, unpublished report exactly
(mean weight 1.519271).

Raw text format (1-indexed), all row-aligned with ``*_A.txt``::

    <NAME>_A.txt                 "u, v"  one directed pair per line (both dirs listed)
    <NAME>_graph_indicator.txt   graph id of each node
    <NAME>_node_labels.txt       label of each node
    <NAME>_edge_labels.txt       label of each edge (per _A.txt line)
    <NAME>_edge_attributes.txt   attribute(s) of each edge (per _A.txt line)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from ..core.graph import Graph

WEIGHT_MIN = 0.5
WEIGHT_MAX = 5.0


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _edge_distance(attr_line: str) -> float:
    parts = [float(x) for x in attr_line.split(",")]
    return abs(parts[0]) if len(parts) == 1 else math.hypot(*parts)


def _norm_token(line: str) -> str:
    """Canonical categorical token for a label line (single- or multi-column)."""
    return ",".join(p.strip() for p in line.split(","))


def convert(raw_dir: str | Path, name: str) -> dict:
    base = Path(raw_dir) / name
    graph_of_node = [int(x) for x in _read_lines(base / f"{name}_graph_indicator.txt")]
    # Labels are treated as categorical tokens (the raw line), so multi-column labels
    # (e.g. Cuneiform's node labels) are handled the same as single-integer labels.
    node_token = [_norm_token(ln) for ln in _read_lines(base / f"{name}_node_labels.txt")]
    edge_pairs = [tuple(int(x) for x in ln.split(",")) for ln in _read_lines(base / f"{name}_A.txt")]
    edge_token = [_norm_token(ln) for ln in _read_lines(base / f"{name}_edge_labels.txt")]
    edge_dist = [_edge_distance(ln) for ln in _read_lines(base / f"{name}_edge_attributes.txt")]

    # Per-dataset min-max scaling of raw distances to [WEIGHT_MIN, WEIGHT_MAX] (D7).
    d_min, d_max = min(edge_dist), max(edge_dist)
    span = d_max - d_min

    def weight(d: float) -> float:
        if span == 0.0:
            return WEIGHT_MIN
        return WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * (d - d_min) / span

    # Remap node/edge label tokens to contiguous ids; keep node tokens as label_strings.
    unique_labels = sorted(set(node_token))
    label_index = {tok: i for i, tok in enumerate(unique_labels)}
    edge_label_index = {tok: i for i, tok in enumerate(sorted(set(edge_token)))}

    # Global 1-indexed node id -> (graph id, local 0-based id).
    per_graph_nodes: dict[int, list[int]] = defaultdict(list)
    for global_id, gid in enumerate(graph_of_node, start=1):
        per_graph_nodes[gid].append(global_id)
    local_of: dict[int, tuple[int, int]] = {}
    for gid, node_ids in per_graph_nodes.items():
        for local, global_id in enumerate(node_ids):
            local_of[global_id] = (gid, local)

    # Undirected edges, deduplicated: keep one entry per {u, v}.
    per_graph_edges: dict[int, dict[tuple[int, int], tuple[int, float]]] = defaultdict(dict)
    for line, (u, v) in enumerate(edge_pairs):
        gu, lu = local_of[u]
        gv, lv = local_of[v]
        if gu != gv or lu == lv:
            continue
        key = (lu, lv) if lu < lv else (lv, lu)
        if key not in per_graph_edges[gu]:
            per_graph_edges[gu][key] = (edge_label_index[edge_token[line]], weight(edge_dist[line]))

    graphs_json = []
    for gid in sorted(per_graph_nodes):
        node_ids = per_graph_nodes[gid]
        vertices = [
            {"id": local, "label": label_index[node_token[global_id - 1]]}
            for local, global_id in enumerate(node_ids)
        ]
        edges = [
            {"src": a, "dst": b, "label": el, "weight": w}
            for (a, b), (el, w) in sorted(per_graph_edges[gid].items())
        ]
        graphs_json.append({"id": gid - 1, "vertices": vertices, "edges": edges})

    return {
        "name": name.lower(),
        "description": f"{name} from TUDataset. Edge weights: per-dataset min-max of "
                       f"edge distance to [{WEIGHT_MIN}, {WEIGHT_MAX}] (see DesignDecisions D7).",
        "label_strings": list(unique_labels),
        "graphs": graphs_json,
    }


def convert_and_save(raw_dir: str | Path, name: str, out_path: str | Path) -> Path:
    data = convert(raw_dir, name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    return out


def load_graphs(raw_dir: str | Path, name: str) -> list[Graph]:
    """Convert and return graphs directly, without writing JSON."""
    data = convert(raw_dir, name)
    graphs = []
    for gj in data["graphs"]:
        node_labels = {v["id"]: v["label"] for v in gj["vertices"]}
        edges = [(e["src"], e["dst"], e["label"], e["weight"]) for e in gj["edges"]]
        graphs.append(Graph.of(node_labels, edges, graph_id=gj["id"]))
    return graphs
