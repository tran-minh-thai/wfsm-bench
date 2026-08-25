# wfsm-bench

**A benchmark for weighted frequent subgraph mining: eight search methods, one objective,
and a comparison that is verified to be fair.**

Weighted frequent subgraph mining is posed here as a discrete optimization problem: find the
Top-K connected subgraphs maximizing a fitness that combines weighted support (WSUP),
average total weight (ATW), and a size penalty. What the repository provides is not one
method but a level field for comparing several.

Eight searches share the same objective, archive, validity rules and result-selection rule,
so the only thing that differs is the search strategy:

| Search | Kind |
|---|---|
| `exhaustive` | complete enumeration — the true optimum wherever it is tractable |
| `greedy` | beam search, optionally capped to a budget |
| `pso` | discrete particle swarm over graph-edit moves |
| `clpso` | comprehensive-learning particle swarm |
| `multiswarm` | dynamic multi-swarm particle swarm |
| `hybrid` | swarm started from a shallow beam frontier |
| `aco` | ant colony — constructive rather than position-based |
| `genetic` | elitist genetic algorithm |
| `random` | blind sampling, the sanity floor |

Having the exhaustive optimum available is what makes the numbers meaningful: a gap is
measured against the real answer, not against the best result anyone happened to get.

## Fair comparison, made checkable

Comparing metaheuristics is easy to get wrong in ways that look fine. Three habits in this
repository exist because each of them caught a defect that had produced a wrong conclusion.

**Budgets count distinct patterns scored, not calls.** Matching a pattern against the
database is the expensive step, and a cache makes revisiting one free. Charging per call
gives a method that wanders back over old ground far less real search than one that caches,
for the same nominal budget. Every search scores through one cached evaluator
(`pso.optimizer.Evaluator`) so the unit of work is the same for all of them.

**A search that stalls restarts rather than stops.** Left to stop, most methods refuse a
larger budget entirely — several were found scoring the same number of distinct patterns
whether given 180 evaluations or 5400. Comparing them at a large budget was comparing
several identical small searches.

**Invariants are tested, not assumed.** `tests/test_search_invariants.py` runs the same
checks against every method: that it spends a meaningful share of its budget, that different
seeds take different routes, that one seed reproduces, that more budget never hurts, and that
an unreachable budget still terminates. A search can return valid patterns, never beat the
optimum and reproduce per seed while barely searching at all, and only these checks notice.

## Requirements

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip`. To set up and run everything in one go there, use
[`run_all_windows.bat`](run_all_windows.bat); for the same steps one at a time, see
[docs/WINDOWS_STEPS.txt](docs/WINDOWS_STEPS.txt).

Dependencies: `networkx` (subgraph matching), `numpy` (randomness and statistics), `pyyaml`
(configuration), `matplotlib` (figures), `pytest` (tests).

## Datasets

Two edge-weighted TUDataset collections: **BZR_MD** (306 dense graphs) and **Cuneiform**
(267 sparse graphs). Fetch and convert them once:

```bash
.venv/bin/python scripts/prepare_datasets.py
```

This writes `data/BZR_MD.json` and `data/Cuneiform.json`. Edge weights are the TUDataset edge
distances, min-max scaled per dataset to `[0.5, 5.0]`.

A tiny hand-checked dataset in `data/sample/` is what the tests run against.

## Running the experiments

Each script writes CSVs to `results/` and figures to `results/figures/`.

| Command | What it produces |
|---|---|
| `python scripts/run_budget_regimes.py` | **the main experiment** — every method across four budgets |
| `python scripts/run_weight_mode.py` | is the objective degenerate? the two weight modes, and the shape of the landscape each produces |
| `python scripts/run_pareto.py` | Pareto-front quality: hypervolume ratio and coverage |
| `python scripts/run_diversity_sweep.py` | how much reported diversity is the search and how much the admission rule |
| `python scripts/run_full_scale.py` | methods across maximum pattern size, parallelised |
| `python scripts/run_methods.py` | all methods side by side at one budget |
| `python scripts/run_size_sweep.py` | the same, across pattern sizes |
| `python scripts/run_experiments.py` | swarm results, scalability, pivot-strategy ablation |
| `python scripts/run_comparison.py` | the swarm against the exhaustive optimum, by dataset size |
| `python scripts/run_budget_sweep.py` | how the swarm's gap closes as its budget grows |
| `python scripts/make_figures.py` | re-renders figures from existing CSVs |
| `python scripts/make_tables.py` | re-renders the result tables from existing CSVs |

Start with the budget sweep, since a single budget can invert the ranking:

```bash
python scripts/run_budget_regimes.py --preset quick --workers 8      # a few minutes
python scripts/run_budget_regimes.py --preset balanced --workers 10  # a few hours
python scripts/run_budget_regimes.py --preset full --workers 10      # overnight
```

Independent runs are spread across processes, so `--workers` should roughly match the core
count; results do not depend on it, because every run is independently seeded. Start scripts
from a file rather than an interactive prompt — worker processes re-import the main module.

## Reproducibility

- Every stochastic run takes an explicit seed; experiments repeat over several seeds and
  report mean and standard deviation.
- All tunable constants live in `src/wfsmbench/config.py` (`Params`) and can be loaded from
  YAML. Settings whose meaning depends on the run length say so where they are defined.
- The objective does not depend on the search configuration. The fitness normalizers are
  anchored on `Params.size_reference`, a constant chosen once for a study, never on
  `max_edges`, which bounds the search space. An anchor that moves with the bound would make
  the same pattern score differently at every size, so a sweep over maximum pattern size
  would compare different objectives — set `size_reference` to the largest `max_edges` the
  study uses and leave it there. `Params` rejects a smaller value.
- Matching uses node and edge labels only; edge weights enter during scoring, never during
  matching.
- Reported tables are generated from the CSVs by `scripts/make_tables.py`, never typed by
  hand. Hand-transcribed tables drift: the ones this replaced still had the deterministic
  columns right and every stochastic column wrong, from a run several commits old.

## Tests

```bash
.venv/bin/python -m pytest -q
```

157 checks cover the graph model and canonical signatures, the dataset loaders,
preprocessing, the weight measures, subgraph matching, pivot scoring, the graph-edit moves,
each search, the multi-objective machinery, the experiment harness, and the cross-method
invariants above. Expected values are hand-computed on the sample dataset, and the BZR_MD
loader is checked against the published dataset statistics.

## Layout

```
src/wfsmbench/
  core/         graph model, particle, canonical signature
  io/           dataset readers, TUDataset conversion
  preprocess/   dataset statistics, frequent edge templates
  measure/      W, ATW, WSUP, fitness, diversity, Pareto metrics
  match/        VF2 subgraph matching
  pivot/        pivot scoring and selection strategies
  pso/          graph-edit moves, the swarm loop and its variants
  baselines/    exhaustive, greedy, ant colony, genetic, random
  experiments/  runners, parallel execution, figures
  config.py     all tunable parameters
scripts/        experiment drivers
tests/          test suite, including the cross-method invariants
data/sample/    tiny hand-checked dataset used by the tests
```

## Notes

- Design notes under `docs/` are written in Vietnamese; the code and this guide are in
  English. `DesignDecisions.md` records every choice made where the method description left
  something open, with the reasoning — read it before changing the measures or the search.
- Manuscripts and third-party reference papers are not part of this repository.
- Generated datasets, result CSVs and figures are not tracked; regenerate them with the
  scripts above.
