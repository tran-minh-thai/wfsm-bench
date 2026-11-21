"""Runs for the supplementary selector experiments T1 and T2 of docs/SUPPLEMENTARY_PROTOCOL.md.

T1 gives every fixed rule the selector's cost, once as one long run per paying seed at
b_eq = b_d + M b_r (arm t1a) and once as k runs at b_d with fresh seeds, best kept (t1b).
T2 runs successive halving on the racing seeds at a starting budget chosen so that the total
does not exceed the selector's cost. The survivor and the k-th restart are paid with the
recorded runs at b_d; only the runs no artefact holds are made here.

Every run goes through scripts/run_weight_mode.py::_job, the function that produced the
recorded runs, with the configuration's own graph limit, rho and pruning flag read from those
runs. Before any new run is made, a sample of recorded runs is repeated and the script
refuses if one fitness differs, so the new rows are comparable with the old ones.

Fitness is a deterministic count; no timing is written.

    python scripts/run_selector_supplement.py            # all phases, resumable
    python scripts/run_selector_supplement.py --plan     # print the run counts only
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_weight_mode as W                      # noqa: E402  the runner of the recorded runs
from run_selector import ALL_PAIRS_RUNS          # noqa: E402  the selector table's run files
from wfsmbench.io import load_dataset            # noqa: E402
from wfsmbench.preprocess import build_statistics  # noqa: E402

OUT = ROOT / "results-invariant" / "revision-2026-11-21"
RUNS = OUT / "selector_supplement_runs.csv"
M = 7
HEADER = ["dataset", "graph_limit", "max_size", "min_support", "weight_mode", "prune", "rho",
          "budget", "method", "seed", "fitness"]
EXTRA_SEED_BASE = 1000


# ----------------------------------------------------------------------------- recorded runs

def recorded_rows() -> list[dict]:
    paths = sorted({p for pattern in ALL_PAIRS_RUNS for p in glob.glob(str(ROOT / pattern))})
    if not paths:
        raise SystemExit("[refused] none of the selector table's run files exists")
    out = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            out.extend(csv.DictReader(handle))
    return out


def configurations(rows: list[dict]) -> dict[tuple, dict]:
    """(dataset, mode, K, support) -> graph_limit, rho, prune, and the recorded fitness table."""
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["dataset"], r["weight_mode"], int(r["max_size"]), int(r["min_support"]))
        info = out.setdefault(key, {"graph_limit": set(), "rho": set(), "prune": set(),
                                    "table": defaultdict(lambda: defaultdict(dict))})
        info["graph_limit"].add(int(r["graph_limit"]))
        info["rho"].add(float(r["rho"]))
        info["prune"].add(r["prune_infrequent"] == "1")
        info["table"][int(r["budget"])][r["method"]][int(r["seed"])] = float(r["fitness"])
    for key, info in out.items():
        for field in ("graph_limit", "rho", "prune"):
            if len(info[field]) != 1:
                raise SystemExit(f"[refused] configuration {key} was run with {field} "
                                 f"{sorted(info[field])}; the supplement cannot match it")
            info[field] = info[field].pop()
    return out


def cells(configs: dict[tuple, dict]) -> list[dict]:
    """The selector table's cells: every budget pair, the seeds present at both budgets for every
    method, race on the even ones, pay on the odd ones (scripts/run_selector.py::_score)."""
    out = []
    for key in sorted(configs):
        table = configs[key]["table"]
        budgets = sorted(table)
        for i, cheap in enumerate(budgets):
            for dear in budgets[i + 1:]:
                methods = sorted(set(table[cheap]) & set(table[dear]))
                seeds = sorted(set.intersection(*[set(table[b][m]) for b in (cheap, dear)
                                                  for m in methods])) if methods else []
                if len(seeds) < 4 or len(methods) < 2:
                    continue
                race = [s for s in seeds if s % 2 == 0]
                pay = [s for s in seeds if s % 2 == 1]
                if len(race) != len(pay):
                    raise SystemExit(f"[refused] {key} {cheap}->{dear}: {len(race)} racing seeds "
                                     f"against {len(pay)} paying; the protocol assumes n_r = n_d")
                out.append({"config": key, "cheap": cheap, "dear": dear, "methods": methods,
                            "race": race, "pay": pay})
    return out


# ---------------------------------------------------------------------------------- designs

def b_equal(cell: dict) -> int:
    return cell["dear"] + M * cell["cheap"]


def restarts(cell: dict) -> int:
    return 1 + (M * cell["cheap"]) // cell["dear"]


def extra_seed(pay_seed: int, j: int) -> int:
    if not 1 <= j <= 9:
        raise SystemExit(f"[refused] restart index {j} outside 1..9 would collide with another seed")
    return EXTRA_SEED_BASE + 10 * pay_seed + j


def halving_schedule(cell: dict) -> tuple[int, list[int], int]:
    """(b_0, the round budgets, the realized total cost) for successive halving at the
    selector's cost; refuses if no b_0 >= 1 fits."""
    n, cheap, dear = len(cell["race"]), cell["cheap"], cell["dear"]
    keep = [M]
    while keep[-1] > 1:
        keep.append(math.ceil(keep[-1] / 2))
    budget_cap = M * cheap
    best = None
    for b0 in range(1, cheap + 1):
        rounds = [min(b0 * 2 ** j, dear) for j in range(len(keep) - 1)]
        spent = sum(m * b for m, b in zip(keep[:-1], rounds))
        if spent <= budget_cap:
            best = (b0, rounds, n * spent + n * dear)
    if best is None:
        raise SystemExit(f"[refused] no starting budget fits the selector's cost in {cell['config']}")
    return best


def halving_survivors(cell: dict, fitness) -> tuple[list[list[str]], str | None]:
    """The methods alive at each round and the winner, or (rounds so far, None) when a round's
    runs are missing. `fitness(budget, method, seed)` returns a float or None."""
    _b0, rounds, _cost = halving_schedule(cell)
    alive = list(cell["methods"])
    history = []
    for budget in rounds:
        history.append(list(alive))
        means = {}
        for m in alive:
            values = [fitness(budget, m, s) for s in cell["race"]]
            if any(v is None for v in values):
                return history, None
            means[m] = sum(values) / len(values)
        alive = sorted(alive, key=lambda m: (-means[m], m))[:math.ceil(len(alive) / 2)]
    return history, alive[0]


# ------------------------------------------------------------------------------------ runs

def run_key(config: tuple, budget: int, method: str, seed: int) -> tuple:
    return (*config, budget, method, seed)


def existing_runs() -> dict[tuple, float]:
    out = {}
    if RUNS.exists():
        with RUNS.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                out[(r["dataset"], r["weight_mode"], int(r["max_size"]), int(r["min_support"]),
                     int(r["budget"]), r["method"], int(r["seed"]))] = float(r["fitness"])
    return out


def run_jobs(jobs: list[tuple], configs: dict, workers: int) -> None:
    """jobs: (config, budget, method, seed). Written one row at a time; skips what exists."""
    done = existing_runs()
    todo = sorted({j for j in jobs if run_key(*j) not in done},
                  key=lambda j: (j[0], -j[1], j[2], j[3]))
    if not todo:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fresh = not RUNS.exists()
    with RUNS.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if fresh:
            writer.writerow(HEADER)
        by_dataset = defaultdict(list)
        for job in todo:
            by_dataset[job[0][0]].append(job)
        for dataset, group in sorted(by_dataset.items()):
            limits = {configs[j[0]]["graph_limit"] for j in group}
            if len(limits) != 1:
                raise SystemExit(f"[refused] {dataset} runs at graph limits {sorted(limits)}")
            graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
            subset = graphs[:limits.pop()]
            stats = build_statistics(subset)
            specs = []
            for config, budget, method, seed in group:
                info = configs[config]
                specs.append((method, config[2], budget, config[1], config[3], seed,
                              info["prune"], info["rho"], 0.0))
            print(f"  {dataset}: {len(specs)} runs", flush=True)
            with ProcessPoolExecutor(max_workers=workers, initializer=W._init,
                                     initargs=(subset, stats)) as pool:
                for (config, budget, method, seed), out in zip(group, pool.map(W._job, specs,
                                                                                chunksize=2)):
                    info = configs[config]
                    writer.writerow([dataset, info["graph_limit"], config[2], config[3], config[1],
                                     int(info["prune"]), info["rho"], budget, method, seed,
                                     f"{out[6]:.6f}"])
                    handle.flush()


def check_reproduction(configs: dict, workers: int) -> int:
    """Repeat one recorded run per (configuration, method) and refuse on any difference."""
    sample = []
    for config in sorted(configs):
        table = configs[config]["table"]
        budget = max(table)
        for method in sorted(table[budget]):
            seed = sorted(table[budget][method])[-1]
            sample.append((config, budget, method, seed, table[budget][method][seed]))
    mismatches = []
    by_dataset = defaultdict(list)
    for item in sample:
        by_dataset[item[0][0]].append(item)
    for dataset, group in sorted(by_dataset.items()):
        info0 = configs[group[0][0]]
        graphs, _ = load_dataset(ROOT / "data" / f"{dataset}.json")
        subset = graphs[:info0["graph_limit"]]
        stats = build_statistics(subset)
        specs = [(m, c[2], b, c[1], c[3], s, configs[c]["prune"], configs[c]["rho"], 0.0)
                 for c, b, m, s, _f in group]
        with ProcessPoolExecutor(max_workers=workers, initializer=W._init,
                                 initargs=(subset, stats)) as pool:
            for (c, b, m, s, want), out in zip(group, pool.map(W._job, specs)):
                if f"{out[6]:.6f}" != f"{want:.6f}":
                    mismatches.append((c, b, m, s, want, out[6]))
    print(f"[reproduction] {len(sample)} recorded runs repeated, {len(mismatches)} differ", flush=True)
    if mismatches or not sample:
        for item in mismatches[:10]:
            print(f"  [mismatch] {item}")
        raise SystemExit("[refused] recorded runs do not reproduce here; no new run is written")
    return len(sample)


# ------------------------------------------------------------------------------- provenance

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_provenance(argv: list[str], extra: dict) -> None:
    import networkx
    import numpy
    import scipy
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts"],
                           capture_output=True, text=True).stdout.strip()
    data = {"command": ["python", "scripts/run_selector_supplement.py", *argv],
            "commit": commit, "tree": "clean" if not dirty else f"MODIFIED ({len(dirty.splitlines())} files)",
            "script_sha256": sha256(Path(__file__)),
            "runner_sha256": sha256(ROOT / "scripts" / "run_weight_mode.py"),
            "python": platform.python_version(), "numpy": numpy.__version__,
            "networkx": networkx.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "data_sha256": {p.name: sha256(p) for p in sorted((ROOT / "data").glob("*.json"))},
            **extra}
    (OUT / "selector_supplement_runs.provenance.json").write_text(
        json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="")


# -------------------------------------------------------------------------------------- main

def plan(cell_list: list[dict]) -> dict[str, list[tuple]]:
    t1a = [(c["config"], b_equal(c), m, s) for c in cell_list for m in c["methods"] for s in c["pay"]]
    k_max: dict[tuple, int] = {}
    for c in cell_list:
        for s in c["pay"]:
            key = (c["config"], c["dear"], s)
            k_max[key] = max(k_max.get(key, 1), restarts(c))
    methods = {c["config"]: c["methods"] for c in cell_list}
    t1b = [(config, dear, m, extra_seed(s, j)) for (config, dear, s), k in sorted(k_max.items())
           for m in methods[config] for j in range(1, k)]
    return {"t1a": t1a, "t1b": t1b}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the run counts and stop")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    configs = configurations(recorded_rows())
    cell_list = cells(configs)
    jobs = plan(cell_list)
    print(f"[cells] {len(cell_list)} cells in {len(configs)} configurations")
    for arm, items in jobs.items():
        print(f"[plan] {arm}: {len(set(items))} runs, budgets {sorted({j[1] for j in items})}")
    schedules = {(c["config"], c["cheap"], c["dear"]): halving_schedule(c) for c in cell_list}
    print(f"[plan] t2: starting budgets {sorted({s[0] for s in schedules.values()})}, "
          f"round budgets {sorted({b for s in schedules.values() for b in s[1]})}")
    if args.plan:
        return 0

    reproduced = check_reproduction(configs, args.workers)
    write_provenance(argv, {"reproduced_runs": reproduced})

    # T2 first, one round at a time: each round's runs depend on the previous round's ranking.
    for round_index in range(3):
        done = existing_runs()
        def lookup(config):
            return lambda b, m, s: done.get(run_key(config, b, m, s))
        needed = []
        for c in cell_list:
            history, winner = halving_survivors(c, lookup(c["config"]))
            if winner is not None:
                continue
            _b0, rounds, _cost = halving_schedule(c)
            alive = history[-1]
            budget = rounds[len(history) - 1]
            needed += [(c["config"], budget, m, s) for m in alive for s in c["race"]]
        print(f"[t2] round {round_index + 1}: {len(set(needed))} runs", flush=True)
        run_jobs(needed, configs, args.workers)
    print("[t1b]", flush=True)
    run_jobs(jobs["t1b"], configs, args.workers)
    print("[t1a]", flush=True)
    run_jobs(jobs["t1a"], configs, args.workers)
    write_provenance(argv, {"reproduced_runs": reproduced, "rows": len(existing_runs())})
    print(f"[done] {RUNS.relative_to(ROOT).as_posix()}: {len(existing_runs())} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
