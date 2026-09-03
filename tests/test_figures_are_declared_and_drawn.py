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
DRAFT = ROOT.parent / "paper" / "ban_thao.tex"
# The manuscript pulls each figure in by its own name and states the source artefact through a
# generated command, so the PNG filenames live in the generated file rather than in the prose.
# Checking only the prose would look clean while naming nothing.
SO_LIEU = ROOT.parent / "paper" / "bang" / "so_lieu.tex"

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
    # A builder that declares nothing would report every declared figure as present.
    declared = _declared()
    assert len(declared) >= 4, (
        f"the builder declares {len(declared)} figure(s); the manuscript carries four, so "
        f"this check would pass over almost nothing")
    absent = [name for name in declared
              if not (FIGS / name).exists() or (FIGS / name).stat().st_size < 5000]
    assert not absent, (
        f"{len(absent)} declared figure(s) missing or truncated: {sorted(absent)}\n"
        f"  run scripts/make_figures.py")


@needs_manuscript("figure_labels.vi.json")
def test_no_image_in_the_figures_directory_is_orphaned() -> None:
    r"""A picture nothing builds any more predates the current results and will still be read.

    SUA 2026-09-02: glob every file, not *.png. Restricting by extension makes the check blind
    to exactly the file someone adds next -- a leftover .pdf or .svg sits in the directory,
    gets picked up by \includegraphics, which resolves by extension order rather than by which
    file is current, and is read as a fresh figure. The check also asserts the directory is not
    empty, since an empty directory has no orphans and reports a pass.
    """
    declared = set(_declared())
    tep = sorted(p for p in FIGS.iterdir() if p.is_file() and not p.name.startswith("."))
    assert len(tep) >= 4, (
        f"only {len(tep)} file(s) in {FIGS}; with none there this check finds no orphans and "
        f"says so as if it had looked")
    orphans = sorted(p.name for p in tep if p.name not in declared)
    assert not orphans, (
        f"{len(orphans)} image(s) no builder produces: {orphans}\n"
        f"  delete them, or add the builder back -- a stale picture is read as a fresh one")


@pytest.mark.skipif(not (DRAFT.exists() and SO_LIEU.exists()),
                    reason="the draft is outside this repository")
def test_the_draft_refers_to_each_figure_it_should() -> None:
    # LaTeX escapes the underscore in a filename, so compare with the escapes removed rather
    # than with the raw text: matching the raw text reported all four figures as unnamed while
    # every one of them was in the file.
    text = (DRAFT.read_text(encoding="utf-8")
            + SO_LIEU.read_text(encoding="utf-8")).replace("\\_", "_")
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


# --------------------------------------------------------- Vietnamese labels on every figure

LABELS = ROOT.parent / "paper" / "figure_labels.vi.json"


def _builders() -> dict[str, str]:
    """{figure filename: source of the function that draws it}, read with ast.

    By name would be guesswork: the builders are not named after their outputs.
    """
    import ast
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    than = {node.name: ast.get_source_segment(text, node)
            for node in tree.body if isinstance(node, ast.FunctionDef)}
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: than[build.__name__] for name, build in module.FIGURES}


@needs_manuscript("figure_labels.vi.json")
def test_every_figure_draws_its_labels_from_the_label_file() -> None:
    """A builder that never asks for labels renders in English, and nothing says so.

    The builder refuses when a KEY is missing, which looks like enough. It is not: the basin
    figure carried English title and axes for as long as it existed because its function never
    called labels() at all, so the refusal never ran. Measured 2026-09-03: three of four
    figures asked for labels, the fourth did not, and the label file simply had no key for it.
    Comparing keys against keys would not have found that -- the absent key was consistent
    with the absent call. So this asks the other question: does every DECLARED figure ask?
    """
    builders = _builders()
    assert len(builders) >= 4, f"only {len(builders)} builder(s) found; the parse is wrong"
    cam = sorted(name for name, than in builders.items() if "labels(" not in than)
    assert not cam, (
        f"{len(cam)} of {len(builders)} figure(s) never ask the label file for anything, so "
        f"they render in whatever language their source was written in: {cam}")


@needs_manuscript("figure_labels.vi.json")
def test_no_label_key_is_orphaned_and_none_is_missing() -> None:
    """Both directions. An unused key is a figure someone deleted; a missing one is English."""
    import json
    import re
    keys = {k for k in json.loads(LABELS.read_text(encoding="utf-8")) if not k.startswith("_")}
    asked = set()
    for than in _builders().values():
        asked |= set(re.findall(r'labels\(\s*["\']([^"\']+)["\']\s*\)', than))
    assert asked, "no builder asks for a label key; the parse is wrong"
    thieu = sorted(asked - keys)
    thua = sorted(keys - asked)
    assert not thieu, f"builders ask for keys the label file does not have: {thieu}"
    assert not thua, (
        f"the label file carries keys no builder asks for: {thua}. Either a figure was "
        f"deleted and its labels stayed, or a builder stopped asking.")
