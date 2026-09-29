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


@pytest.fixture(scope="session")
def drawn_figures() -> Path:
    """The figures directory, drawn once from the tracked results if any figure is missing.

    results/figures/ is not tracked, so a fresh clone has no figures; the checks that read the
    drawn files draw them first instead of skipping.
    """
    import importlib.util
    import subprocess
    import sys
    script = ROOT / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures_for_tests", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    figures = ROOT / "results" / "figures"
    if not all((figures / name).exists() for name, _ in module.FIGURES):
        done = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                              capture_output=True, text=True)
        assert done.returncode == 0, f"the figure builder failed\n{done.stdout}{done.stderr}"
    return figures


def needs_manuscript(*names: str):
    missing = [n for n in names if not (PAPER / n).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"outside this repository: {', '.join(missing)} (expected in {PAPER})")
