"""Every figure the builder produces must exist, and must be referred to by the manuscript.

Two ways a figure goes wrong quietly. It can fail to redraw and leave the previous run's
picture behind wearing today's date -- the builder deletes before drawing to stop that, and
this checks the deletion did not simply leave a hole. And it can be drawn but never cited,
which the editorial rules forbid: a figure no sentence points at is a figure the reader has
no reason to look at.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import needs_manuscript

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"
FIGS = ROOT / "results" / "figures"
DRAFT = ROOT.parent / "paper" / "ban_thao.md"

# Figures the draft must cite. The other builders serve sections not yet written, so they are
# listed here rather than required of the draft -- and listing them is what makes adding a
# third unreferenced figure a visible decision instead of an accident.
CITED_BY_THE_DRAFT = {"front_position_vs_bound.png", "regret_vs_race_size.png",
                      "quantum_sweep.png", "basin_vs_aco.png"}


def _declared() -> list[str]:
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [name for name, _build in module.FIGURES]


def test_the_builder_declares_the_figures_the_draft_cites() -> None:
    declared = set(_declared())
    missing = CITED_BY_THE_DRAFT - declared
    assert not missing, f"the draft cites figures nothing builds: {sorted(missing)}"


@needs_manuscript("figure_labels.vi.json")
def test_every_declared_figure_is_on_disk_and_not_empty() -> None:
    absent = [name for name in _declared()
              if not (FIGS / name).exists() or (FIGS / name).stat().st_size < 5000]
    assert not absent, (
        f"{len(absent)} declared figure(s) missing or truncated: {sorted(absent)}\n"
        f"  run scripts/make_figures.py")


@needs_manuscript("figure_labels.vi.json")
def test_no_image_in_the_figures_directory_is_orphaned() -> None:
    """A picture nothing builds any more predates the current results and will still be read."""
    declared = set(_declared())
    orphans = sorted(p.name for p in FIGS.glob("*.png") if p.name not in declared)
    assert not orphans, (
        f"{len(orphans)} image(s) no builder produces: {orphans}\n"
        f"  delete them, or add the builder back -- a stale picture is read as a fresh one")


@pytest.mark.skipif(not DRAFT.exists(), reason="the draft is outside this repository")
def test_the_draft_refers_to_each_figure_it_should() -> None:
    text = DRAFT.read_text(encoding="utf-8")
    unreferenced = sorted(name for name in CITED_BY_THE_DRAFT if name not in text)
    assert not unreferenced, (
        f"the draft never names {unreferenced}; a figure no sentence points at gives the "
        f"reader no reason to look at it")


@needs_manuscript("figure_labels.vi.json")
def test_the_builder_runs_and_reproduces_every_figure(tmp_path: Path) -> None:
    """A builder that has not been run since its inputs changed proves nothing about them."""
    done = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, f"the figure builder failed\n{done.stdout}{done.stderr}"
    for name in _declared():
        assert (FIGS / name).exists(), f"{name} was declared but not written"
