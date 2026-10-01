"""Shared fixtures, and the condition for the few checks that reach outside this repository.

The manuscript lives in a sibling directory that is deliberately not part of this repository.
A machine that clones only the code does not have it, so the checks that compare the figure
list with the manuscript skip there. Everything else runs from a clone alone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper"


def result_tree_state(root: Path = ROOT) -> dict[str, tuple[int, int]]:
    """{path: (size, modification time)} for every file under the result trees, ignored ones
    included: the publisher on the measurement machine reports new files there."""
    state = {}
    for tree in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("results")):
        for path in tree.rglob("*"):
            if path.is_file():
                info = path.stat()
                state[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns)
    return state


@pytest.fixture(scope="session", autouse=True)
def result_trees_untouched():
    """The test suite must leave the result trees exactly as it found them.

    A test that writes there is invisible on the development machine and noisy on the
    measurement machine, where the suite runs right before each stage and the publisher then
    reports every file it finds. It happened on 2026-09-29 with the figures.
    """
    before = result_tree_state()
    yield
    after = result_tree_state()
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    assert not changed, (f"the test run wrote {len(changed)} file(s) under the result trees: "
                         f"{changed[:10]}")


@pytest.fixture(scope="session")
def drawn_figures(tmp_path_factory) -> Path:
    """A directory holding every figure, drawn once per test session from the tracked results.

    Drawn into a temporary directory, not results/figures/: a test run must not write into the
    result trees, where the measurement machine's publisher reports every new untracked file.
    """
    import subprocess
    import sys
    figures = tmp_path_factory.mktemp("figures")
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"),
                           "--out", str(figures)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, f"the figure builder failed\n{done.stdout}{done.stderr}"
    return figures


def needs_manuscript(*names: str):
    missing = [n for n in names if not (PAPER / n).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"outside this repository: {', '.join(missing)} (expected in {PAPER})")
