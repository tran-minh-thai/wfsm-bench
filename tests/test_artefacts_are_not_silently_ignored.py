"""Every artefact a runner writes must be able to reach the repository.

`.gitignore` carried `results/pareto_*.csv` from a runner generation that was deleted. The
glob outlived its producer and swallowed `pareto_size_*.csv`, the 63-row summary of a 6.2
hour run, without any message: the run reported `[ok]`, the file existed on disk, and it
simply never appeared in a commit. Excluding by name goes stale exactly when something new
is added.

This asks the question by property instead: for each runner that writes into results/, take
the name it writes and check git does not ignore it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Runner -> a representative artefact name it writes, taken from the f-string in its source.
WRITES = {
    "run_pareto_size.py": "results/pareto_size_Cuneiform.csv",
    "run_basins.py": "results/basins_cuneiform.csv",
    "run_degeneracy.py": "results/degeneracy_frontier_cuneiform.csv",
    "run_selector.py": "results/selector_regret.csv",
}


def ignored(relative: str) -> bool:
    """--no-index is required. Without it git skips the rules for files already tracked,
    so a check on a tracked artefact passes for the wrong reason and proves nothing --
    three of the four checks below were vacuous until this was measured."""
    done = subprocess.run(["git", "check-ignore", "--no-index", "-q", relative], cwd=ROOT)
    return done.returncode == 0


@pytest.mark.parametrize("runner,artefact", sorted(WRITES.items()))
def test_the_artefact_a_runner_writes_is_not_ignored(runner: str, artefact: str) -> None:
    assert (SCRIPTS / runner).exists(), f"{runner} is gone; drop its row from WRITES"
    assert not ignored(artefact), (
        f"{runner} writes {artefact}, and .gitignore excludes it. The run will report [ok] "
        f"and the artefact will never reach a commit."
    )


def _a_still_ignored_path() -> str | None:
    """A path matching some rule that is currently in force under results/.

    Found from the rules rather than named here. The first version hard-coded one pattern as
    its canary; when that rule was removed for a good reason the canary went with it, and the
    test failed for a reason that had nothing to do with what it guards.
    """
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        rule = line.strip()
        if not rule.startswith("results") or rule.startswith("#") or rule.startswith("!"):
            continue
        candidate = rule.replace("**/", "").replace("*", "canary")
        if candidate.endswith("/"):
            candidate += "canary.txt"
        if ignored(candidate):
            return candidate
    return None


def test_the_check_can_actually_fail() -> None:
    """A guard that never fires proves nothing: something must still be ignored."""
    candidate = _a_still_ignored_path()
    assert candidate is not None, (
        "no rule under results/ is in force any more, so the check above cannot distinguish "
        "a working .gitignore from an empty one"
    )
    print(f"\n[canary] {candidate} is ignored, so the check can fire")
