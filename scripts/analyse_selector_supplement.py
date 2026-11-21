"""T1, T2 and T3 of docs/SUPPLEMENTARY_PROTOCOL.md, from the recorded and supplementary runs.

Per cell: the gap to the exhaustive optimum, gap = 100 (F* - F) / |F*|, and the regret against
the best method at b_d, for the racing selector, every fixed rule at b_d, at the selector's
cost as one long run (a) and as restarts with fresh seeds (b), and successive halving at the
selector's cost. Per configuration: the mean over its budget pairs. Then the selector against
each rival over the 16 configurations: mean difference with a bootstrap interval and a
Holm-corrected Wilcoxon signed-rank test, with and without the Cuneiform configurations.

Reads results/ and results-invariant/revision-2026-11-21/selector_supplement_runs.csv; writes
three CSV files beside the latter. Refuses on a missing run rather than averaging fewer.

    python scripts/analyse_selector_supplement.py
"""

from __future__ import annotations

import csv
import glob
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_selector_supplement as R   # noqa: E402

BOOTSTRAP, BOOT_SEED, ALPHA = 10_000, 20261121, 0.05


def optima() -> dict[tuple, float]:
    """(dataset, mode, K, support) -> F*, from every landscape artefact; refuses on disagreement."""
    seen: dict[tuple, set[float]] = {}
    for path in sorted(glob.glob(str(ROOT / "results" / "weight_mode_landscape*.csv"))):
        with open(path, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r.get("fitness_quantum", "0") not in ("0", "") or r.get("graph_offset", "0") not in ("0", ""):
                    continue
                key = (r["dataset"], r["weight_mode"], int(r["max_size"]), int(r["min_support"]))
                seen.setdefault(key, set()).add(float(r["optimum_fitness"]))
    mixed = {k: v for k, v in seen.items() if len(v) != 1}
    if mixed:
        raise SystemExit(f"[refused] landscape artefacts disagree on F*: {mixed}")
    return {k: v.pop() for k, v in seen.items()}


def gap(fstar: float, value: float) -> float:
    return 100.0 * (fstar - value) / abs(fstar)


def regret(best: float, value: float) -> float:
    return 100.0 * (best - value) / best


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main() -> int:
    configs = R.configurations(R.recorded_rows())
    cell_list = R.cells(configs)
    extra = R.existing_runs()
    fstar = optima()
    picks = {}
    with (ROOT / "results" / "selector_regret_allpairs.csv").open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["seed_split"] == "disjoint":
                picks[(r["dataset"], r["weight_mode"], int(r["max_size"]), int(r["min_support"]),
                       int(r["race_budget"]), int(r["pay_budget"]))] = r["picked"]

    def fitness(config, budget, method, seed):
        table = configs[config]["table"]
        if seed in table.get(budget, {}).get(method, {}):
            return table[budget][method][seed]
        value = extra.get(R.run_key(config, budget, method, seed))
        if value is None:
            raise SystemExit(f"[refused] missing run {config} budget {budget} {method} seed {seed}; "
                             f"run scripts/run_selector_supplement.py")
        return value

    cell_rows = []
    for c in cell_list:
        config, cheap, dear, pay, race = c["config"], c["cheap"], c["dear"], c["pay"], c["race"]
        if config not in fstar:
            raise SystemExit(f"[refused] no exhaustive optimum for {config}")
        f_star = fstar[config]
        mean_pay = {m: statistics.mean(fitness(config, dear, m, s) for s in pay) for m in c["methods"]}
        best_bd = max(mean_pay.values())
        # The race, recomputed and checked against the selector table's own pick.
        race_means = {m: statistics.mean(fitness(config, cheap, m, s) for s in race) for m in c["methods"]}
        pick = max(race_means, key=race_means.get)
        recorded = picks.get((*config, cheap, dear))
        if recorded != pick:
            raise SystemExit(f"[refused] {config} {cheap}->{dear}: the race picks {pick}, the "
                             f"selector table says {recorded}")
        row = {"dataset": config[0], "weight_mode": config[1], "max_size": config[2],
               "min_support": config[3], "race_budget": cheap, "pay_budget": dear, "n": len(pay),
               "t_race": R.M * len(race) * cheap + len(pay) * dear, "f_star": f"{f_star:.6f}",
               "picked": pick, "gap_selector": gap(f_star, mean_pay[pick]),
               "regret_selector": regret(best_bd, mean_pay[pick])}
        b_eq, k = R.b_equal(c), R.restarts(c)
        for m in c["methods"]:
            long_run = statistics.mean(fitness(config, b_eq, m, s) for s in pay)
            best_of_k = statistics.mean(
                max([fitness(config, dear, m, s)] + [fitness(config, dear, m, R.extra_seed(s, j))
                                                    for j in range(1, k)]) for s in pay)
            row[f"gap_bd_{m}"] = gap(f_star, mean_pay[m])
            row[f"gap_a_{m}"] = gap(f_star, long_run)
            row[f"gap_b_{m}"] = gap(f_star, best_of_k)
            row[f"regret_a_{m}"] = regret(best_bd, long_run)
            row[f"regret_b_{m}"] = regret(best_bd, best_of_k)
        b0, rounds, cost = R.halving_schedule(c)
        _history, winner = R.halving_survivors(c, lambda b, m, s, cf=config: fitness(cf, b, m, s))
        if winner is None:
            raise SystemExit(f"[refused] successive halving incomplete in {config} {cheap}->{dear}")
        row.update({"b_eq": b_eq, "restarts": k, "halving_b0": b0,
                    "halving_rounds": "|".join(map(str, rounds)), "halving_cost": cost,
                    "halving_winner": winner, "gap_halving": gap(f_star, mean_pay[winner]),
                    "regret_halving": regret(best_bd, mean_pay[winner])})
        if cost > row["t_race"]:
            raise SystemExit(f"[refused] halving costs {cost} > {row['t_race']} in {config}")
        cell_rows.append(row)

    methods = sorted({m for c in cell_list for m in c["methods"]})
    columns = [k for k in cell_rows[0] if k.startswith(("gap_", "regret_"))]
    by_config: dict[tuple, list[dict]] = {}
    for row in cell_rows:
        by_config.setdefault((row["dataset"], row["weight_mode"], row["max_size"], row["min_support"]),
                             []).append(row)
    config_rows = [{"dataset": k[0], "weight_mode": k[1], "max_size": k[2], "min_support": k[3],
                    "cells": len(v), **{col: statistics.mean(r[col] for r in v) for col in columns}}
                   for k, v in sorted(by_config.items())]

    rules = [f"gap_{v}_{m}" for v in ("a", "b") for m in methods]
    best_by_cell = min(rules, key=lambda col: statistics.mean(r[col] for r in cell_rows))
    best_by_config = min(rules, key=lambda col: statistics.mean(r[col] for r in config_rows))

    stats_rows = []
    for scope, subset in (("all", config_rows),
                          ("without_cuneiform", [r for r in config_rows if r["dataset"] != "Cuneiform"])):
        rivals = ([f"gap_bd_{m}" for m in methods] + rules
                  + [best_by_cell, "gap_halving"])
        labels = ([f"fixed {m} at b_d" for m in methods] + [f"fixed {m}, one long run" for m in methods]
                  + [f"fixed {m}, restarts" for m in methods]
                  + [f"best fixed rule at equal cost ({best_by_cell[4:]})", "successive halving"])
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.integers(0, len(subset), size=(BOOTSTRAP, len(subset)))
        pending = []
        for rival, label in zip(rivals, labels):
            diff = np.array([r[rival] - r["gap_selector"] for r in subset])
            boot = diff[draws].mean(axis=1)
            nonzero = int(np.count_nonzero(diff))
            p = float(wilcoxon(diff).pvalue) if nonzero else 1.0
            pending.append({"scope": scope, "configurations": len(subset), "rival": label,
                            "mean_gap_rival": float(np.mean([r[rival] for r in subset])),
                            "mean_gap_selector": float(np.mean([r["gap_selector"] for r in subset])),
                            "mean_difference": float(diff.mean()),
                            "ci_low": float(np.percentile(boot, 2.5)),
                            "ci_high": float(np.percentile(boot, 97.5)),
                            "nonzero_differences": nonzero, "selector_better": int((diff > 0).sum()),
                            "selector_worse": int((diff < 0).sum()), "p_wilcoxon": p})
        for row, adjusted in zip(pending, holm([r["p_wilcoxon"] for r in pending])):
            row["p_holm"] = adjusted
            stats_rows.append(row)

    out = R.OUT
    for name, data in (("selector_supplement_cells.csv", cell_rows),
                       ("selector_supplement_configurations.csv", config_rows),
                       ("selector_supplement_statistics.csv", stats_rows)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            for r in data:
                writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[cells] {len(cell_rows)}; best fixed rule at equal cost: {best_by_cell} over cells, "
          f"{best_by_config} over configurations")
    for r in stats_rows:
        print(f"  {r['scope']:<18} {r['rival']:<52} diff {r['mean_difference']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p {r['p_wilcoxon']:.4f} holm {r['p_holm']:.4f} "
              f"(+{r['selector_better']}/-{r['selector_worse']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
