# wfsm-bench

Weighted frequent subgraph mining posed as optimisation: find the connected subgraphs that
maximise a fitness combining weighted support (WSUP), average total weight (ATW) and a size
penalty. The repository holds the objective and its variants, eight search methods and an
exhaustive enumerator that gives the true optimum wherever it is tractable, a racing selector
that picks a method per dataset, the experiment stages behind every reported number, and the
artefacts those stages produced.

| method | kind |
|---|---|
| `exhaustive` | complete enumeration, the reference optimum |
| `pso` | discrete particle swarm over graph-edit moves |
| `clpso` | comprehensive-learning particle swarm |
| `multiswarm` | dynamic multi-swarm particle swarm |
| `aco` | ant colony optimisation, constructive |
| `genetic` | elitist genetic algorithm |
| `local_search` | iterated local search |
| `random` | blind sampling, the floor |
| `greedy` | beam search |

Every search scores patterns through one cached evaluator, and budgets count distinct patterns
scored, so methods are compared on equal work. A search that stalls restarts part of its
population instead of stopping early. `tests/test_search_invariants.py` checks, for every
method, that it uses its budget, that seeds differ and reproduce, and that a larger budget
never does worse.

## Repository layout

```
src/wfsmbench/        the library
  core/                 graph model, particle, canonical signature
  io/                   dataset readers and converters (TUDataset, molecular, protein)
  preprocess/           dataset statistics, frequent edge templates
  measure/              weight measures, fitness, diversity, Pareto metrics, selector regret
  match/                subgraph matching (VF2 monomorphism)
  pivot/                pivot scoring and selection
  pso/                  graph-edit moves and the swarm variants
  baselines/            exhaustive, greedy, ant colony, genetic, local search, random, gSpan
  experiments/          run harness and parallel execution
  config.py             every tunable parameter (Params)
scripts/              experiment drivers, table and figure builders, run-record tools
tests/                test suite
data/                 prepared datasets (built by scripts/prepare_datasets.py)
  sample/               tiny hand-checked dataset the tests use
  pdb_contact.json      the one dataset that cannot be rebuilt; tracked
figures/labels.en.json  every word drawn inside a figure
results/              artefacts whose numbers depend on the machine (times) or on seeds
  runs/                 a copy of every published run, never overwritten
  submitted/            the artefacts of a submitted version, never overwritten
results-invariant/    artefacts that are counts or set comparisons, machine-independent
docs/DesignDecisions.md  every modelling choice left open by the method description
run_experiments_windows.bat  the stage driver (the one definition of every stage)
run_all_windows.bat   set up a Windows machine and run stages, publishing each one
COMMIT_MAP.tsv        old commit -> new commit, for records written before 2026-09-29
```

## Requirements

- Python 3.11 or newer. The recorded runs used Python 3.12.10 on Windows (AMD64); the
  development machine uses Python 3.14 on macOS (arm64).
- The packages in `requirements.txt`: `networkx`, `numpy`, `scipy`, `pyyaml`, `matplotlib`,
  `pytest`. The recorded runs used numpy 2.5.1, networkx 3.6.1, scipy 1.18.0, matplotlib
  3.11.1 and pyyaml 6.0.3. Search results depend on the numpy and networkx versions (numpy
  gives no bit-stream guarantee across releases); exhaustive results do not.
- Redrawing figures needs the Times New Roman font; the builder stops if it is missing.
- No GPU. The work is CPU-bound subgraph matching, spread over worker processes.

## Setup

There is nothing to compile. Create an environment and install the packages.

macOS / Linux:

```bash
git clone https://github.com/tran-minh-thai/wfsm-bench.git
cd wfsm-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (Command Prompt):

```bat
git clone https://github.com/tran-minh-thai/wfsm-bench.git
cd wfsm-bench
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, `run_all_windows.bat` does all of this, and then runs stages.

## Datasets

```bash
python scripts/prepare_datasets.py
```

downloads the TUDataset collections (https://www.chrsmrrs.com/graphkerneldatasets) into
`data/raw/` and writes one JSON file per dataset into `data/`. It skips a dataset whose raw
files are already present. Edge weights are scaled per dataset to `[0.5, 5.0]`
(`docs/DesignDecisions.md`, decision D7). The molecular `_bond` datasets are rebuilt from
the real bonds, with the bond length as the weight. `pdb_contact.json` is tracked in `data/`
because the protein structures it was built from are no longer available.

The measured statistics of every dataset, from `results-invariant/dataset_profile.csv`:

<!-- generated:datasets -->
| dataset | graphs | mean nodes | mean edges | node labels | edge labels | mean weight | source |
|---|---:|---:|---:|---:|---:|---:|---|
| BZR_MD | 306 | 21.3 | 225.1 | 8 | 5 | 1.519 | TUDataset, edge distances |
| BZR_bond | 405 | 35.8 | 38.4 | 10 | 1 | 2.084 | TUDataset, rebuilt from bonds and atom coordinates |
| COX2_bond | 467 | 41.2 | 43.4 | 8 | 1 | 2.239 | TUDataset, rebuilt from bonds and atom coordinates |
| Cuneiform | 267 | 21.3 | 44.8 | 12 | 2 | 0.786 | TUDataset, edge attributes |
| DHFR_bond | 756 | 42.4 | 44.5 | 9 | 1 | 1.664 | TUDataset, rebuilt from bonds and atom coordinates |
| QM9_bond | 5000 | 14.2 | 14.2 | 5 | 4 | 2.393 | TUDataset, rebuilt from bonds and atom coordinates |
| alchemy_full_bond | 5000 | 10.0 | 10.3 | 6 | 4 | 0.811 | TUDataset, rebuilt from bonds and atom coordinates |
| pdb_contact | 473 | 83.6 | 162.1 | 6 | 2 | 3.267 | protein contact graphs (tracked, not rebuilt) |
<!-- /generated:datasets -->

Every run record (see *Run records*) carries a SHA-256 of each file in `data/`, so a run on
rebuilt data can be told apart from a run on the original files.

## Tests

```bash
python -m pytest -q
```

Some checks skip on a clone: those that compare the figures and tables with the manuscript,
which is not part of this repository, and those that read the raw downloads in `data/raw/`,
which exist only after `scripts/prepare_datasets.py` has been run.

## A small end-to-end run (about four minutes)

```bash
python scripts/prepare_datasets.py
python scripts/run_weight_mode.py --preset quick --modes sum --seeds 3 --workers 4 --suffix _smoke
python scripts/compare_landscape.py --suffix _smoke
```

The run enumerates seven cells of Cuneiform (60 graphs, patterns up to 4 edges) and runs the
seven search methods on them. Enumeration draws no random numbers, so the last command checks
that the pool size, optimum and plateau counts of every cell match the tracked artefacts
exactly; it fails if any differs or if nothing could be compared. The `_smoke` files are
ignored by git.

## Running the experiments

The experiments are organised in 20 stages. `run_experiments_windows.bat` is the single
definition of what each stage runs.

On Windows:

```bat
run_experiments_windows.bat 17          REM one stage
run_experiments_windows.bat 3 4 5       REM several
run_all_windows.bat 17                  REM set up or update the checkout, test, run, publish
run_all_windows.bat rerun-untraceable   REM every stage whose artefacts have no traceable record
```

Before each stage the driver refuses to run from a checkout that is behind its remote or has
uncommitted changes under `src/` or `scripts/`, then writes the run record
`results/environment_stage<N>.txt`. After each stage `scripts/publish_results.py` keeps a copy
of what the stage wrote in `results/runs/`, commits the artefacts and pushes them.
`scripts/stages_to_rerun.py` lists the stages whose artefacts cannot be traced to a commit in
this history.

On macOS or Linux, run the same commands from the repository root with the environment active.
`python scripts/stage_commands.py <N>` prints the commands of stage N, read from the Windows
driver; they are listed below. Set `WORKERS` to the number of cores first
(`export WORKERS=8`); the worker count does not change any result, because every run is seeded
on its own.

| stage | what it measures | approximate time | depends on the machine |
|---:|---|---|---|
| 1 | calibration of the discount rho on QM9_bond | 15 min | no |
| 2 | search methods on QM9_bond | 30 min | seeds only |
| 3 | size degeneracy on Cuneiform, sizes 4 and 5, and its rho calibration | 20 min | no |
| 4 | search methods on Cuneiform, size 5 | 25 min | seeds only |
| 5 | anti-monotone pruning on and off, size 6 | 1 h | seeds only |
| 6 | the landscape of all 21 Cuneiform cells, both weight modes | 7 h | no |
| 7 | search methods on Cuneiform, size 6 | 5.5 h | seeds only |
| 8 | search methods on Cuneiform, size 4, looser budget | 10 min | seeds only |
| 9 | search methods on BZR_bond, COX2_bond, DHFR_bond | 2 h | seeds only |
| 10 | plateau structure forced by rounding the objective, QM9_bond | 1.7 h | seeds only |
| 11 | basins of steepest ascent, six datasets | 30 min | no |
| 12 | search methods on alchemy_full_bond | 40 min | seeds only |
| 13 | the racing selector on disjoint race and pay seeds, six datasets | 2.5 h | seeds only |
| 14 | determinism: stage 13 against the earlier runs of the same cells | 10 s | no |
| 15 | rho calibration on four more datasets | 15 min | no |
| 16 | where Pareto-front selectors land on the size axis, Cuneiform | 6.5 h | no |
| 17 | the same on the five `_bond` datasets | 3.2 h | no |
| 18 | gSpan against the exhaustive enumerator, timed | 3 h | yes (times) |
| 19 | gSpan against the enumerator, pattern sets only | 7 h | no |
| 20 | enumeration cost at the third size bound, timed | 9 to 11 h | yes (times) |

Times are the estimates in the header of `run_experiments_windows.bat`, for a 12-thread
machine. "Seeds only" means the numbers depend
on the seeds and the numpy version but not on the machine; "no" means they are exact counts
that any machine reproduces; "yes" means they are wall-clock times.

<!-- generated:stage-commands -->
**Stage 1**

```bash
python scripts/run_degeneracy.py --dataset QM9_bond --limit 200 --sizes 4,5 --supports 2 --calibrate-rho --suffix _qm9 > results/qm9_rho.log 2>&1
```

**Stage 2**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset QM9_bond)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --seeds 20 --workers $WORKERS --suffix _qm9 > results/qm9_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_qm9.csv > results/qm9_statistics.log 2>&1
```

**Stage 3**

```bash
python scripts/run_degeneracy.py --dataset Cuneiform --limit 60 --sizes 4,5 --calibrate-rho --suffix _cuneiform > results/cuneiform_degeneracy.log 2>&1
```

**Stage 4**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset Cuneiform)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --sizes 5 --supports 2 --modes "$MODES" --seeds 20 --workers $WORKERS --suffix _s5 > results/s5_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_s5.csv > results/s5_statistics.log 2>&1
```

**Stage 5**

```bash
python scripts/run_weight_mode.py --preset deep --sizes 6 --supports 30 --modes sum --seeds 20 --workers $WORKERS --suffix _noprune > results/noprune.log 2>&1
python scripts/run_weight_mode.py --preset deep --sizes 6 --supports 30 --modes sum --seeds 20 --workers $WORKERS --prune --suffix _prune > results/prune.log 2>&1
```

**Stage 6**

```bash
python scripts/run_weight_mode.py --preset deep --supports 2,6,12,18,30,36,42 --modes sum,per_edge --landscape-only > results/landscape_sweep.log 2>&1
```

**Stage 7**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset Cuneiform)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --sizes 6 --supports 2 --modes "$MODES" --seeds 20 --workers $WORKERS --suffix _s6 > results/s6_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_s6.csv > results/s6_statistics.log 2>&1
```

**Stage 8**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset Cuneiform)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --sizes 4 --supports 2 --modes "$MODES" --seeds 20 --workers $WORKERS --suffix _s4loose > results/s4loose.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_s4loose.csv > results/s4loose_statistics.log 2>&1
```

**Stage 9**

```bash
for D in BZR COX2 DHFR; do
  MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset ${D}_bond)" ] && MODES=sum,discounted
  python scripts/run_weight_mode.py --preset deep --dataset ${D}_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --seeds 20 --workers $WORKERS --suffix _${D}bond > results/${D}bond_methods.log 2>&1
  python scripts/run_statistics.py --runs results/weight_mode_runs_${D}bond.csv > results/${D}bond_statistics.log 2>&1
done
```

**Stage 10**

```bash
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0 --suffix _q000 > results/q000_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q000.csv > results/q000_statistics.log 2>&1
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0.0002 --suffix _q002 > results/q002_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q002.csv > results/q002_statistics.log 2>&1
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0.0005 --suffix _q005 > results/q005_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q005.csv > results/q005_statistics.log 2>&1
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0.001 --suffix _q010 > results/q010_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q010.csv > results/q010_statistics.log 2>&1
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0.002 --suffix _q020 > results/q020_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q020.csv > results/q020_statistics.log 2>&1
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0.005 --suffix _q050 > results/q050_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q050.csv > results/q050_statistics.log 2>&1
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes sum --seeds 20 --workers $WORKERS --quantum 0.01 --suffix _q100 > results/q100_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_q100.csv > results/q100_statistics.log 2>&1
```

**Stage 11**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset QM9_bond)" ] && MODES=sum,discounted
python scripts/run_basins.py --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --suffix _qm9 > results/basins_qm9.log 2>&1
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset alchemy_full_bond)" ] && MODES=sum,discounted
python scripts/run_basins.py --dataset alchemy_full_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --suffix _alch > results/basins_alch.log 2>&1
for D in BZR COX2 DHFR; do
  MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset ${D}_bond)" ] && MODES=sum,discounted
  python scripts/run_basins.py --dataset ${D}_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --suffix _${D}bond > results/basins_${D}bond.log 2>&1
done
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset Cuneiform)" ] && MODES=sum,discounted
python scripts/run_basins.py --dataset Cuneiform --limit 60 --sizes 5 --supports 2 --modes "$MODES" --suffix _cuneiform > results/basins_cuneiform.log 2>&1
```

**Stage 12**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset alchemy_full_bond)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --dataset alchemy_full_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --seeds 20 --workers $WORKERS --suffix _alch > results/alch_methods.log 2>&1
python scripts/run_statistics.py --runs results/weight_mode_runs_alch.csv > results/alch_statistics.log 2>&1
```

**Stage 13**

```bash
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset Cuneiform)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --dataset Cuneiform --limit 60 --sizes 5 --supports 2 --modes "$MODES" --budgets 60,240 --seeds 40 --workers $WORKERS --suffix _selCuneiform > results/selCuneiform_methods.log 2>&1
MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset QM9_bond)" ] && MODES=sum,discounted
python scripts/run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 --sizes 5 --supports 2 --modes "$MODES" --budgets 60,240 --seeds 40 --workers $WORKERS --suffix _selQM9_bond > results/selQM9_bond_methods.log 2>&1
for D in BZR_bond COX2_bond DHFR_bond alchemy_full_bond; do
  MODES=sum; [ -n "$(python scripts/pick_stable_rho.py --dataset ${D})" ] && MODES=sum,discounted
  python scripts/run_weight_mode.py --preset deep --dataset ${D} --limit 200 --sizes 5 --supports 2 --modes "$MODES" --budgets 60,240 --seeds 40 --workers $WORKERS --suffix _sel${D} > results/sel${D}_methods.log 2>&1
done
python scripts/run_selector.py --runs "results/weight_mode_runs_sel*.csv" > results/selector.log 2>&1
python scripts/run_selector.py --all-pairs --out results/selector_regret_allpairs.csv > results/selector_allpairs.log 2>&1
```

**Stage 14**

```bash
rm -f results/determinism_stage14.log
echo "=== selQM9_bond vs qm9 ===" >> results/determinism_stage14.log
python scripts/check_determinism.py --a selQM9_bond --b qm9 --families weight_mode_runs >> results/determinism_stage14.log 2>&1
echo "=== selBZR_bond vs BZRbond ===" >> results/determinism_stage14.log
python scripts/check_determinism.py --a selBZR_bond --b BZRbond --families weight_mode_runs >> results/determinism_stage14.log 2>&1
echo "=== selCOX2_bond vs COX2bond ===" >> results/determinism_stage14.log
python scripts/check_determinism.py --a selCOX2_bond --b COX2bond --families weight_mode_runs >> results/determinism_stage14.log 2>&1
echo "=== selDHFR_bond vs DHFRbond ===" >> results/determinism_stage14.log
python scripts/check_determinism.py --a selDHFR_bond --b DHFRbond --families weight_mode_runs >> results/determinism_stage14.log 2>&1
echo "=== selalchemy_full_bond vs alch ===" >> results/determinism_stage14.log
python scripts/check_determinism.py --a selalchemy_full_bond --b alch --families weight_mode_runs >> results/determinism_stage14.log 2>&1
echo "=== selCuneiform vs s5 ===" >> results/determinism_stage14.log
python scripts/check_determinism.py --a selCuneiform --b s5 --families weight_mode_runs >> results/determinism_stage14.log 2>&1
```

**Stage 15**

```bash
for D in BZR_bond COX2_bond DHFR_bond alchemy_full_bond; do
  python scripts/run_degeneracy.py --dataset ${D} --limit 200 --sizes 4,5 --supports 2 --calibrate-rho --suffix _${D} > results/rho_${D}.log 2>&1
done
python scripts/pick_stable_rho.py --report > results/rho_summary.txt 2>&1
```

**Stage 16**

```bash
python scripts/run_pareto_size.py --dataset Cuneiform --graph-limit 60 --sizes 4 --suffix _Cuneiform_s4 > results/pareto_size_s4.log 2>&1
python scripts/run_pareto_size.py --dataset Cuneiform --graph-limit 60 --sizes 4 5 --suffix _Cuneiform_s45 > results/pareto_size_s45.log 2>&1
python scripts/run_pareto_size.py --dataset Cuneiform --graph-limit 60 --sizes 4 5 6 --suffix _Cuneiform > results/pareto_size_Cuneiform.log 2>&1
```

**Stage 17**

```bash
for D in BZR_bond COX2_bond DHFR_bond QM9_bond alchemy_full_bond; do
  python scripts/run_pareto_size.py --dataset ${D} --graph-limit 200 --sizes 3 4 5 6 --supports 0.01 0.0333333 0.10 0.20 0.30 0.50 0.60 0.70 --suffix _${D} > results/pareto_size_${D}.log 2>&1
done
```

**Stage 18**

```bash
for D in Cuneiform; do
  python scripts/run_gspan_baseline.py --dataset ${D} --graph-limit 60 --sizes 3 4 5 --support 2 --suffix _${D} > results/gspan_${D}.log 2>&1
done
for D in BZR_bond COX2_bond DHFR_bond QM9_bond alchemy_full_bond; do
  python scripts/run_gspan_baseline.py --dataset ${D} --graph-limit 200 --sizes 4 5 --support 2 --suffix _${D} > results/gspan_${D}.log 2>&1
done
python scripts/run_gspan_baseline.py --dataset Cuneiform --graph-limit 60 --sizes 6 --support 2 --skip-exhaustive --suffix _Cuneiform_s6 > results/gspan_Cuneiform_s6.log 2>&1
```

**Stage 19**

```bash
python scripts/run_gspan_baseline.py --dataset Cuneiform --graph-limit 60 --sizes 6 --support 2 --counts-only --suffix _full_Cuneiform > results/gspan_agree_Cuneiform_s6.log 2>&1
```

**Stage 20**

```bash
for D in BZR_bond COX2_bond DHFR_bond QM9_bond alchemy_full_bond; do
  python scripts/run_gspan_baseline.py --dataset ${D} --graph-limit 200 --sizes 6 --support 2 --suffix _${D} >> results/gspan_${D}.log 2>&1
done
python scripts/run_gspan_baseline.py --dataset Cuneiform --graph-limit 60 --sizes 6 --support 2 --suffix _Cuneiform >> results/gspan_Cuneiform.log 2>&1
```
<!-- /generated:stage-commands -->

## Tables and figures

```bash
python scripts/make_tables.py --out results/tables.md
python scripts/make_figures.py
```

Both read only the tracked artefacts. Figures are written to `results/figures/` (not tracked)
as vector PDF, in Times New Roman, with every string between 9 and 10 pt at the width each
figure is printed at (`PRINTED_FRACTION` in `scripts/make_figures.py`); the words come from
`figures/labels.en.json`.

## Results

`results/` holds artefacts that depend on seeds or on the machine; `results-invariant/` holds
exact counts and set comparisons that any machine reproduces. The files at the top of each
tree are the current version, and a later run of the same stage replaces them.
`results/runs/<UTC time>_<commit>_stage<N>/` keeps every published run as it was, and
`scripts/freeze_submitted.py --label <name>` copies the artefacts of a submitted version to
`results/submitted/`; neither is ever overwritten.

| files | written by | contents |
|---|---|---|
| `results/weight_mode_landscape*.csv` | `run_weight_mode.py` | one row per cell (dataset, graphs, size bound, support, weight mode): pool size, optimum, its size and support, plateau and local-optimum counts |
| `results/weight_mode_runs*.csv` | `run_weight_mode.py` | one row per search run: cell, budget, method, seed, fitness, whether it found the optimum |
| `results/weight_mode_methods*.csv` | `run_weight_mode.py` | per cell, budget and method: mean and standard deviation of fitness, gap to the optimum, hits |
| `results/statistics_pairwise_*.csv` | `run_statistics.py` | Wilcoxon signed-rank tests for every pair of methods, paired by seed, Holm-corrected |
| `results/degeneracy_frontier_*.csv` | `run_degeneracy.py` | per pattern size: the best pattern's support and weight, the terms of the fitness decomposition, measured and modelled fitness |
| `results/degeneracy_gamma_*.csv` | `run_degeneracy.py` | the optimum's size and fitness for each size-penalty coefficient |
| `results/degeneracy_rho*.csv` | `run_degeneracy.py` | the optimum's size by bound for each discount rho, and whether it is stable |
| `results/rho_summary.txt` | `pick_stable_rho.py` | the calibrated rho per dataset, or why there is none |
| `results/basins_*.csv` | `run_basins.py` | basins of steepest ascent: share of the pool reaching the optimum, largest basin, path lengths |
| `results/selector_regret*.csv` | `run_selector.py` | per cell: the method the race picked, its regret, and the regret of each fixed rule |
| `results/selector_seed_sweep.csv` | `run_selector_seed_sweep.py` | regret of the race by race size, against the transfer-only floor |
| `results/pareto_size_*.csv` | `run_pareto_size.py` | the non-dominated front of each cell and where each selector's pick lies on it |
| `results/gspan_baseline_*.csv` | `run_gspan_baseline.py` | gSpan and the enumerator per cell: pattern counts, seconds, speed-up, whether the pattern sets agree |
| `results/determinism_*` | `check_determinism.py` | comparisons of repeated runs of the same cells, with the number of rows compared |
| `results/environment_*.txt` | `environment.py` | run records (see below) |
| `results/*.log` | the stage commands | console output of each command, named in the stage commands |
| `results/tables.md` | `make_tables.py` | the result tables, generated from the artefacts |
| `results-invariant/dataset_profile.csv` | `measure_datasets.py` | the dataset statistics shown above |
| `results-invariant/gspan_agreement_*.csv` | `run_gspan_baseline.py` | pattern-set agreement between gSpan and the enumerator, counts only |
| `results-invariant/bound_locality_d*.csv` | `measure_bound_locality.py` | coverage of a host graph by a ball of each remaining depth |
| `results-invariant/embedding_load.csv` | `measure_embedding_load.py` | embeddings per pattern at each size bound |
| `results-invariant/race_only_signals.csv` | `measure_race_only_signals.py` | quantities a race can observe, against the regret it incurs |
| `results-invariant/check_scope_audit.csv` | `audit_check_scope.py` | which checks in `tests/` recompute from artefacts and which compare with typed values |

## Run records

Every stage writes `results/environment_stage<N>.txt` before it runs, with:
the commit and whether the working tree was clean; the upstream state; Python, numpy, networkx,
scipy, matplotlib and pyyaml versions; the machine name, CPU, core count and memory; the memory
limit and time limit (both "none": Python has no heap flag and no run is time-limited); the
SHA-256 of every dataset file; the stage number and worker count. `python
scripts/environment.py` prints the same record for the current machine. A result without a
record is not used.

The history of `main` was rewritten on 2026-09-29, which changed every commit hash. A record
written before then names an old hash; `COMMIT_MAP.tsv` gives the commit holding the same code
today, and `python scripts/commit_map.py <hash>` looks one up.

## Parameters

Every default below is read from the source. `src/wfsmbench/config.py` (`Params`) holds the
parameters of the objective and the searches; the constants after it are fixed by the scripts.
The discount `rho` is calibrated per dataset (stages 1, 3 and 15) and read by the runners from
the calibration artefacts; `results/rho_summary.txt` lists the values.

<!-- generated:parameters -->
| parameter | default | what it sets |
|---|---|---|
| `alpha` | `1.0` | Fitness F(g) = alpha*Norm(WSUP) + beta*Norm(ATW) - gamma*Pen(g). |
| `beta` | `0.5` | Fitness F(g) = alpha*Norm(WSUP) + beta*Norm(ATW) - gamma*Pen(g). |
| `gamma` | `0.1` | Fitness F(g) = alpha*Norm(WSUP) + beta*Norm(ATW) - gamma*Pen(g). |
| `eta1` | `0.1` | Penalty Pen(g) = eta1*\|V\| + eta2*\|E\|. |
| `eta2` | `0.1` | Penalty Pen(g) = eta1*\|V\| + eta2*\|E\|. |
| `weight_mode` | `'sum'` | Weight aggregation for ATW (see measure.evaluation / measure.fitness). |
| `rho` | `0.5` | The discount factor, used only by weight_mode="discounted", and calibrated per dataset: where the optimum leaves the size bound depends on how fast support falls with ... |
| `fitness_quantum` | `0.0` | Rounds the objective the search navigates onto a grid of this width, leaving the true objective untouched. |
| `size_reference` | `6` | Upper anchor for the ATW / WSUP normalizers, counted in pattern edges (decision D2). |
| `lambda1` | `0.25` | Pivot score S(v) = l1*deg + l2*wdeg + l3*cent + l4*lab. |
| `lambda2` | `0.25` | Pivot score S(v) = l1*deg + l2*wdeg + l3*cent + l4*lab. |
| `lambda3` | `0.25` | Pivot score S(v) = l1*deg + l2*wdeg + l3*cent + l4*lab. |
| `lambda4` | `0.25` | Pivot score S(v) = l1*deg + l2*wdeg + l3*cent + l4*lab. |
| `omega` | `0.7` | PSO velocity update. |
| `c1` | `1.5` | PSO velocity update. |
| `c2` | `1.5` | PSO velocity update. |
| `prune_infrequent` | `False` | Refuse to score a pattern that provably cannot be frequent, using the anti-monotonicity of support: every subgraph of a frequent pattern is frequent, so a pattern ... |
| `swarm_size` | `5` | Search budget. |
| `iterations` | `3` | Search budget. |
| `top_k` | `10` | Search budget. |
| `min_support` | `2` | Search budget. |
| `diversity_weight` | `0.0` | Top-K admission: 0.0 keeps the plain best-fitness set, higher values trade fitness for structural novelty between the reported patterns (see measure.diversity). |
| `num_subswarms` | `4` | Dynamic multi-swarm topology (see pso.multiswarm). |
| `regroup_period` | `2` | Dynamic multi-swarm topology (see pso.multiswarm). |
| `restart_fraction` | `0.5` | What a search does when it stops finding anything new. |
| `max_barren_restarts` | `3` | What a search does when it stops finding anything new. |
| `clpso_refresh_gap` | `1` | CLPSO holds a particle's exemplar fixed until its personal best has failed to improve this many rounds (Liang et al., 2006). |
| `elite_fraction` | `0.05` | Genetic baseline: share of the population carried over unchanged each generation. |
| `aco_alpha` | `1.0` | Ant colony (see baselines.aco): trail vs heuristic exponents, evaporation, colony size. |
| `aco_beta` | `2.0` | Ant colony (see baselines.aco): trail vs heuristic exponents, evaporation, colony size. |
| `aco_evaporation` | `0.1` | Ant colony (see baselines.aco): trail vs heuristic exponents, evaporation, colony size. |
| `num_ants` | `10` | Ant colony (see baselines.aco): trail vs heuristic exponents, evaporation, colony size. |
| `velocity_mode` | `'template'` | How a particle's position is compared (see pso.structural). |
| `max_nodes` | `6` | Structural bounds. |
| `max_edges` | `6` | Structural bounds. |
| `max_velocity` | `3` | Structural bounds. |
| `pivot_strategy` | `'scientific'` | Pivot selection strategy (for the ablation study): one of "scientific" (full S(v)), "degree", "random", "none". |

| constant | file | value | what it sets |
|---|---|---|---|
| `STUDY_SIZE_REFERENCE` | `src/wfsmbench/config.py` | `7` | fitness normalisation anchor (pattern edges) used by every experiment script |
| `WEIGHT_MIN` | `src/wfsmbench/io/tudataset.py` | `0.5` | lower end of the per-dataset edge-weight scaling |
| `WEIGHT_MAX` | `src/wfsmbench/io/tudataset.py` | `5` | upper end of the per-dataset edge-weight scaling |
| `QM9_LIMIT` | `scripts/prepare_datasets.py` | `5000` | graphs kept from QM9 when the dataset is built |
| `ALCHEMY_LIMIT` | `scripts/prepare_datasets.py` | `5000` | graphs kept from alchemy_full when built |
| `CUTOFF` | `src/wfsmbench/io/protein.py` | `5.5` | contact distance (angstrom) for the protein contact graphs |
| `STALL_ROUNDS` | `src/wfsmbench/pso/optimizer.py` | `5` | consecutive rounds without a new pattern before any search restarts part of its population |
| `PERTURBATION` | `src/wfsmbench/baselines/local_search.py` | `2` | random edits applied by an iterated-local-search kick |
| `SWARM` | `scripts/run_weight_mode.py` | `20` | population size of every population-based method |
| `BUDGETS` | `scripts/run_weight_mode.py` | `60, 120, 180, 240` | evaluation budgets (distinct patterns scored) per run |
| `SUPPORT_FRACTIONS` | `scripts/run_weight_mode.py` | `0.03, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7` | support thresholds, as a fraction of the graphs used |
| `RHOS` | `scripts/run_degeneracy.py` | `1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05` | discount factors swept when rho is calibrated |
| `GAMMAS` | `scripts/run_degeneracy.py` | `0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1` | size-penalty coefficients swept by the degeneracy analysis |
| `SUPPORT_FRACTIONS` | `scripts/run_pareto_size.py` | `0.03333, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7` | support thresholds of the Pareto-front runs |
| `STUDY_GRAPH_LIMIT` | `scripts/run_pareto_size.py` | `60` | graphs used from Cuneiform in the Pareto-front runs |
| `ALPHA` | `scripts/run_statistics.py` | `0.05` | significance level of the pairwise tests (Holm-corrected) |
| `FIXED` | `scripts/run_selector.py` | `aco, local_search` | fixed single-method rules the racing selector is compared with |
| `DECOMPOSITION_SIZE` | `src/wfsmbench/measure/selector.py` | `10` | race size at which the regret bound's two terms are separated |
| `TIE` | `scripts/measure_race_only_signals.py` | `0.001` | fitness gap under which two methods count as tied in a race |
| `CD` | `scripts/make_figures.py` | `1.007` | Nemenyi critical difference drawn on the rank figures (7 methods, 80 blocks, alpha 0.05) |
<!-- /generated:parameters -->

## Reproducibility notes

- Every stochastic run takes an explicit seed; the worker count does not change any result.
- The fitness normalisers are anchored on a fixed size (`STUDY_SIZE_REFERENCE`), never on the
  search bound, so the same pattern scores the same under every bound.
- Matching uses node and edge labels only; edge weights enter the fitness, never the matching.
- Tables are generated from the artefacts by `scripts/make_tables.py`, never typed by hand.
- Run scripts from a file, not an interactive prompt: worker processes re-import the main
  module.

## License

MIT; see [LICENSE](LICENSE).
