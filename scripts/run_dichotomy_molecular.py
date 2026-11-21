"""T4 of docs/SUPPLEMENTARY_PROTOCOL.md: the growth dichotomy on the five molecular datasets.

The out-of-sample check of the growth dichotomy compared 14 configurations, all Cuneiform,
because only Cuneiform had a measured per_edge optimum. This enumerates the landscape in sum
and per_edge for the 10 configurations of the molecular frontier artefacts (first 200 graphs,
K = 4 and 5, support 2), through the runner's own enumerate_pool and landscape, and then
scores the prediction from the sum frontier (verification.py, S = 7) against it, per dataset.
It also scores the null "the optimum is always a one-edge pattern" on the same cells and on
the 14 Cuneiform ones.

Counts only; the artefact goes to results-invariant/.

    python scripts/run_dichotomy_molecular.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_weight_mode as W                         # noqa: E402
from wfsmbench.io import load_dataset               # noqa: E402
from wfsmbench.measure import verification as V     # noqa: E402
from wfsmbench.preprocess import build_statistics   # noqa: E402

OUT = ROOT / "results-invariant" / "revision-2026-11-21"
LANDSCAPE = OUT / "dichotomy_molecular_landscape.csv"
SUMMARY = OUT / "dichotomy_molecular_summary.csv"
DATASETS = {"BZR_bond": "BZR_bond", "COX2_bond": "COX2_bond", "DHFR_bond": "DHFR_bond",
            "QM9_bond": "qm9", "alchemy_full_bond": "alchemy_full_bond"}
LIMIT, SUPPORT, SIZES, MODES = 200, 2, (4, 5), ("sum", "per_edge")


def frontier(stem: str) -> dict[tuple[int, int], list[dict]]:
    path = ROOT / "results" / f"degeneracy_frontier_{stem}.csv"
    by: dict[tuple[int, int], list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if int(r["graph_limit"]) != LIMIT:
                raise SystemExit(f"[refused] {path.name} holds graph limit {r['graph_limit']}")
            by.setdefault((int(r["max_size"]), int(r["min_support"])), []).append(r)
    return {k: sorted(v, key=lambda r: int(r["edges"])) for k, v in by.items()}


def enumerate_landscapes() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
        subset = graphs[:LIMIT]
        stats = build_statistics(subset)
        for size in SIZES:
            params = W.params_for(size, 240, "sum", SUPPORT)
            pool = W.enumerate_pool(subset, stats, params, f"{dataset} K={size}")
            edges = [c.particle.pattern.num_edges() for c in pool]
            for mode in MODES:
                info = W.landscape(pool, stats, W.params_for(size, 240, mode, SUPPORT))
                rows.append({"dataset": dataset, "graph_limit": LIMIT, "max_size": size,
                             "min_support": SUPPORT, "weight_mode": mode, "pool_size": len(pool),
                             "pool_min_edges": min(edges), "pool_max_edges": max(edges),
                             "optimum_fitness": f"{info['optimum_fitness']:.6f}",
                             "optimum_edges": info["optimum_edges"],
                             "optimum_nodes": info["optimum_nodes"],
                             "best_f_by_edges": info["best_f_by_edges"]})
            print(f"  {dataset} K={size}: pool {len(pool)}", flush=True)
    return rows


def score(rows: list[dict]) -> list[dict]:
    measured = {(r["dataset"], int(r["max_size"]), r["weight_mode"]): r for r in rows}
    out = []
    for dataset, stem in DATASETS.items():
        series_by = frontier(stem)
        for mode in MODES:
            tally = {"dataset": dataset, "weight_mode": mode, "cells": 0, "strict": 0,
                     "predicted": 0, "intermediate": 0, "null_one_edge": 0}
            for size in SIZES:
                series = series_by.get((size, SUPPORT))
                row = measured.get((dataset, size, mode))
                if series is None or row is None:
                    raise SystemExit(f"[refused] no frontier or landscape for {dataset} K={size}")
                optimum = int(row["optimum_edges"])
                tally["cells"] += 1
                tally["strict"] += V.strict_case_holds(V.dichotomy_steps(series, mode), mode)
                tally["predicted"] += V.predicted_argmax(series, mode) == optimum
                tally["intermediate"] += int(row["pool_min_edges"]) < optimum < int(row["pool_max_edges"])
                tally["null_one_edge"] += optimum == 1
            out.append(tally)
    # The same null on the 14 Cuneiform configurations of the existing check.
    cun = [r for r in V.read(ROOT / "results", "weight_mode_landscape.csv")
           if r["weight_mode"] == "per_edge"
           and (int(r["size_bound"]), int(r["min_support"])) in V.frontier_series(ROOT / "results")]
    out.append({"dataset": "Cuneiform", "weight_mode": "per_edge", "cells": len(cun),
                "strict": "", "predicted": "", "intermediate": "",
                "null_one_edge": sum(int(r["optimum_edges"]) == 1 for r in cun)})
    return out


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = enumerate_landscapes()
    # The sum optimum of any configuration an earlier stage enumerated must come out the same.
    recorded = {}
    for path in sorted((ROOT / "results").glob("weight_mode_landscape*.csv")):
        for r in V.read(ROOT / "results", path.name):
            if (r.get("fitness_quantum", "0") in ("0", "") and r["graph_limit"] == str(LIMIT)
                    and r["min_support"] == str(SUPPORT)):
                recorded.setdefault((r["dataset"], int(r["max_size"]), r["weight_mode"]),
                                    set()).add(r["optimum_fitness"])
    compared = [(r, recorded[(r["dataset"], int(r["max_size"]), r["weight_mode"])])
                for r in rows if (r["dataset"], int(r["max_size"]), r["weight_mode"]) in recorded]
    differing = [(r["dataset"], r["max_size"], r["weight_mode"], r["optimum_fitness"], sorted(v))
                 for r, v in compared if v != {r["optimum_fitness"]}]
    print(f"[reproduction] {len(compared)} optima compared with earlier landscapes, "
          f"{len(differing)} differ")
    if not compared or differing:
        raise SystemExit(f"[refused] {differing or 'nothing to compare'}")
    write(LANDSCAPE, rows)
    summary = score(rows)
    write(SUMMARY, summary)
    for r in summary:
        print(f"  {r['dataset']:<18} {r['weight_mode']:<9} cells {r['cells']} strict {r['strict']} "
              f"predicted {r['predicted']} intermediate {r['intermediate']} "
              f"one-edge null {r['null_one_edge']}")
    import networkx
    import numpy
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    (OUT / "dichotomy_molecular.provenance.json").write_text(json.dumps({
        "command": ["python", "scripts/run_dichotomy_molecular.py"], "commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": numpy.__version__,
        "networkx": networkx.__version__, "platform": platform.platform(),
        "size_reference": V.SIZE_REFERENCE}, indent=1) + "\n", encoding="utf-8", newline="")
    print(f"[done] {LANDSCAPE.relative_to(ROOT).as_posix()}, {SUMMARY.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
