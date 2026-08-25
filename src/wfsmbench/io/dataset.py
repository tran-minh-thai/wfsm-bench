"""Minimal reader for the project's own dataset JSON (the sample format).

This is the lightweight loader used from P0 onward. The TUDataset conversion
(edge labels + edge weights for BZR_MD / Cuneiform) is added in P1 and produces the
same JSON schema this reader consumes.

Schema::

    {
      "name": str,
      "label_strings": [str, ...],
      "graphs": [
        {"id": int,
         "vertices": [{"id": int, "label": int}, ...],
         "edges": [{"src": int, "dst": int, "label": int, "weight": float}, ...]}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.graph import Graph


def load_dataset(path: str | Path) -> tuple[list[Graph], list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    label_strings = list(data.get("label_strings", []))
    graphs: list[Graph] = []
    for gj in data["graphs"]:
        node_labels = {v["id"]: v["label"] for v in gj["vertices"]}
        edges = [
            (e["src"], e["dst"], e.get("label", 0), e["weight"])
            for e in gj["edges"]
        ]
        graphs.append(Graph.of(node_labels, edges, graph_id=gj.get("id")))
    return graphs, label_strings
