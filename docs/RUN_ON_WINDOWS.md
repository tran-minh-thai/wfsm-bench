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
```

## 4. Run the experiments

Every experiment runs through one entry point, which writes an environment record beside each
stage's artefacts and stops on the first failure:

```bat
git checkout main
git pull --ff-only
run_experiments_windows.bat            REM the default stages, in result-per-hour order
run_experiments_windows.bat 3 4 5      REM only some
run_experiments_windows.bat 19         REM one stage by number
```

Name the branch before pulling. A bare `git pull` updates whichever branch the machine is
standing on -- which is how a stage once ran three commits behind.

The work moved from `per-edge-objective` to `main` on 2026-09-25, and `per-edge-objective`
was deleted. A checkout still standing on it moves across once, keeping its commits (the two
branches share one history):

```bat
git fetch --prune origin
git checkout -B main per-edge-objective
git branch --set-upstream-to=origin/main
git branch -d per-edge-objective
git pull --ff-only
```

If `git checkout main` instead lands on a `main` older than 2026-08-25, it shares no history
with the remote and `git pull` fails; replace it with `git checkout -B main origin/main`.
`run_all_windows.bat` does both of these by itself, but do them by hand before starting it the
first time: the move rewrites `run_all_windows.bat`, and cmd reads a batch file by byte offset,
so a file rewritten while it runs executes fragments of the new one. Pull first, always; then
the runner's own pull has nothing to change.

The stage table, what each stage supports, and its measured or estimated cost are in the
header comment of `run_experiments_windows.bat`. That file is the source of truth for what
runs; this guide covers only getting the machine ready.

Stages 16, 17 and 19 produce counts and set comparisons rather than seconds, so they do not
depend on the machine and are not in the default list. Stage 18 does measure time and belongs
on this machine and nowhere else.

**Earlier versions of this guide listed three scripts by name that no longer exist**
(`run_full_scale.py`, `run_budget_regimes.py`, `run_diversity_sweep.py`). They were replaced
by the staged entry point above. `tests/test_windows_guide_names_live_scripts.py` now fails if
this file names a script that is not in the repository, because nothing caught the last drift.

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
