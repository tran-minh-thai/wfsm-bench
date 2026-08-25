# Running the experiments on Windows

Guide for running wfsm-bench's experiments on a Windows machine (tested target: Dell XPS,
Intel i7-10750H, 6 cores / 12 threads, 64 GB RAM). No GPU is used — the work is CPU-bound
subgraph matching, parallelised across cores.

## 1. Python environment

Install Python 3.11+ (from python.org), then, in the project folder (`wfsm-bench\`):

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

This installs `networkx`, `numpy`, `pyyaml`, `matplotlib`, `pytest`.

## 2. Datasets

Either copy the already-built `data\BZR_MD.json` and `data\Cuneiform.json` from another
machine into `data\`, or regenerate them (needs internet — downloads TUDataset):

```bat
.venv\Scripts\python scripts\prepare_datasets.py
```

## 3. Check the setup

```bat
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\run_full_scale.py --preset quick
```

The quick preset finishes in seconds and confirms parallelism works on Windows.

## 4. Run the experiments

Two experiments, run in this order. Both overwrite their outputs, so a re-run always
replaces the previous numbers rather than appending to them.

### 4a. Methods vs pattern size

```bat
.venv\Scripts\python scripts\run_full_scale.py --preset balanced --workers 8
```

- `--preset balanced` (default): BZR n=50 sizes 3-5, Cuneiform n=100 sizes 3-6, 5 seeds —
  roughly **30-45 minutes** on this machine, enough numbers for the paper.
- `--preset full --workers 10`: BZR to size 6, Cuneiform to size 7, 10 seeds — an
  **overnight** run for the most complete results.
- `--workers 8` suits 6 cores (leaves thermal headroom on a laptop); use `10` for a
  headless / overnight run.

Outputs: `results\size_<dataset>.csv` and `results\figures\size_<dataset>.png`.

### 4b. Budget regimes (the main experiment)

```bat
.venv\Scripts\python scripts\run_budget_regimes.py --preset balanced --workers 10
```

Sweeps four evaluation budgets (180, 540, 1800 and 5400 distinct patterns scored) across all
six searches on both datasets. This is the experiment that decides how everything else is
read: comparing methods at a single budget can invert the ranking, and several of them stop
being able to use a larger budget at all, which is only visible when the budget is varied.

Check the setup first with `--preset quick`, which is about seven minutes. `--preset
balanced` is roughly **two to three hours** at ten workers; `--preset full` widens the
datasets and pattern sizes and is an overnight run.

Outputs: `results\budget_regimes_<dataset>.csv`, one row per (size, budget, method) giving
the mean fitness, the gap to the exhaustive optimum, and how many seeds reached it.

### 4c. Top-K admission weight (diversity)

```bat
.venv\Scripts\python scripts\run_diversity_sweep.py --preset balanced --workers 8
```

Roughly **10-20 minutes** on the balanced preset. Each (method, seed) is searched once and
re-reported at every weight, so adding weights costs almost nothing.

Outputs: `results\diversity_<dataset>.csv` and `results\figures\diversity_<dataset>.png`.

**Sanity check on the output:** within one method, the `fitness_mean` column must be
identical for every `diversity_weight`. The weight decides only which patterns are
reported, never what the search finds — if that column moves, something is wrong.

## Notes

- **Plug in power and keep it cool.** Sustained all-core matching heats a laptop and can
  thermal-throttle; running fewer workers (8 rather than 12) is often steadier.
- **Memory is not a concern** here (a few GB used; 64 GB is ample).
- **Run from the script, not a REPL.** Windows starts worker processes with `spawn`, which
  re-imports the main module — this only works from a `.py` file that has an
  `if __name__ == "__main__":` guard (the scripts do). Pasting the code into an interactive
  prompt will fail with `BrokenProcessPool`.
- The macOS/Linux virtualenv is **not** portable to Windows — always create a fresh `.venv`
  with step 1.
