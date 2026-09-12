"""Every figure the builder produces must exist, and must be referred to by the manuscript.

Two ways a figure goes wrong quietly. It can fail to redraw and leave the previous run's
picture behind wearing today's date -- the builder deletes before drawing to stop that, and
this checks the deletion did not simply leave a hole. And it can be drawn but never cited,
which the editorial rules forbid: a figure no sentence points at is a figure the reader has
no reason to look at.
"""
from __future__ import annotations

import importlib.util
import re
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


def _declared() -> list[str]:
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [name for name, _build in module.FIGURES]


def _anh_xa_cua_bo_sinh() -> dict:
    """Draft-side figure name -> artefact filename, read from the generator that copies them."""
    spec = importlib.util.spec_from_file_location(
        "sync_manuscript_numbers", DRAFT.parent / "tools" / "sync_manuscript_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.IMAGES)


@pytest.mark.skipif(not DRAFT.exists(), reason="the draft is outside this repository")
def test_the_builder_declares_the_figures_the_draft_uses() -> None:
    """SUA 2026-09-11: derived from the generator instead of a set of filenames kept here.

    The list here went stale the way a list of names always does. It named four figures while
    the draft used six, so two of them were outside every check in this file -- and the two it
    missed were the two added most recently, which is exactly when a hand-kept list is least
    likely to be updated.
    """
    declared = set(_declared())
    dung = set(_anh_xa_cua_bo_sinh().values())
    assert dung, "the generator copies no figure; there is nothing to check"
    missing = sorted(dung - declared)
    assert not missing, (
        f"the draft uses {missing}, which no builder in this repository produces. The file in "
        f"the draft's directory is then whatever was left there, with nothing keeping it in "
        f"step with any artefact.")


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


@pytest.mark.skipif(not DRAFT.exists(), reason="the draft is outside this repository")
def test_every_drawn_figure_is_included_by_the_draft() -> None:
    """SUA 2026-09-11: ask whether the draft INCLUDES the figure, not whether it names the file.

    The old version searched the draft for the source filename, e.g. basin_vs_aco.png. That
    only ever matched because each caption carried a provenance line printing the artefact
    path. The author then required the printed page to carry no trace of the process -- paths
    and script names are how the work was done, not what the figure shows -- those lines went,
    and this check reported four figures as unreferenced while every one of them was in the
    paper, drawn, captioned and pointed at.

    The filename was never the question. The question is whether the draft puts the picture on
    a page, and that is what the include line says. The mapping from artefact to the name the
    draft uses lives in the generator, so it is read from there rather than restated here.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sync_manuscript_numbers", DRAFT.parent / "tools" / "sync_manuscript_numbers.py")
    sinh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sinh)

    than = DRAFT.read_text(encoding="utf-8")
    nhung = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{hinh/([A-Za-z0-9_-]+)\}", than))
    assert nhung, "the draft includes no figure at all; the pattern stopped matching"

    ten_bai = set(sinh.IMAGES)
    assert ten_bai, "the generator copies no figure; there is nothing to check"

    thieu = sorted(ten_bai - nhung)
    assert not thieu, (
        f"the generator copies {thieu} into the draft's figure directory and no include line "
        f"puts {'them' if len(thieu) > 1 else 'it'} on a page; a figure nobody sees is a "
        f"figure that still has to be kept in step with its artefact")

    la = sorted(nhung - ten_bai)
    assert not la, (
        f"the draft includes {la}, which the generator does not copy. Those files are whatever "
        f"was last left in the directory, and nothing keeps them current with any artefact.")


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


# ------------------------------------------- the cost figure refuses a curve through two dots

COT = ("dataset,graph_limit,size_bound,min_support,gspan_patterns,gspan_seconds,"
       "exhaustive_patterns,exhaustive_seconds,speedup,sets_agree,supports_disagreeing,"
       "gspan_optimum_edges,gspan_optimum_fitness")


def _o(dataset: str, bound: int, pool: int, seconds: str) -> str:
    """One cell of the timed enumeration artefact. Only four fields are read here."""
    return f"{dataset},200,{bound},2,{pool},1.000,{pool},{seconds},1.0,1,0,3,0.1"


def _bo_ve(tmp_path: Path, *rows: str):
    """A module bound to a results directory holding exactly the cells named."""
    (tmp_path / "gspan_baseline_probe.csv").write_text(
        "\n".join((COT, *rows)) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("make_figures_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULTS = tmp_path
    return module


@needs_manuscript("figure_labels.vi.json")
def test_the_cost_figure_refuses_a_collection_measured_at_two_bounds(tmp_path: Path) -> None:
    """Two points join into a straight line whatever the truth is.

    This is the refusal the figure exists behind: it could have been drawn at any time from
    the two bounds already on disk, and it would have drawn a growth rate nothing measured.
    The third bound cost nine hours on the measurement machine, so the refusal has to be
    demonstrated rather than asserted -- a guard nobody has seen fail is a guard nobody knows
    is wired up.
    """
    module = _bo_ve(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Beta", 4, 100, "10.0"), _o("Beta", 5, 300, "60.0"),
                    _o("Beta", 6, 900, "400.0"))
    with pytest.raises(SystemExit) as refused:
        module.figure_enumeration_cost(tmp_path / "cost.png")
    assert "Alpha (2)" in str(refused.value), str(refused.value)
    assert "Beta" not in str(refused.value), "the collection with three bounds is not at fault"
    assert not (tmp_path / "cost.png").exists(), "it refused and drew anyway"


@needs_manuscript("figure_labels.vi.json")
def test_the_cost_figure_draws_once_every_collection_has_three(tmp_path: Path) -> None:
    """The other half of the refusal: it must pass on data that earns it."""
    module = _bo_ve(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Alpha", 6, 900, "400.0"))
    module.figure_enumeration_cost(tmp_path / "cost.png")
    assert (tmp_path / "cost.png").stat().st_size > 5000


@needs_manuscript("figure_labels.vi.json")
def test_the_cost_figure_refuses_a_cell_two_artefacts_disagree_on(tmp_path: Path) -> None:
    """One cell, two measurements: the figure cannot pick, so it must not pick silently."""
    module = _bo_ve(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Alpha", 6, 900, "400.0"), _o("Alpha", 6, 900, "700.0"))
    with pytest.raises(SystemExit) as refused:
        module.figure_enumeration_cost(tmp_path / "cost.png")
    assert "measured twice" in str(refused.value), str(refused.value)


@needs_manuscript("figure_labels.vi.json")
def test_the_cost_figure_ignores_a_cell_with_no_enumeration_time(tmp_path: Path) -> None:
    """A gSpan-only cell carries a pool size and no seconds.

    Dividing by a time nobody measured would put a made-up point on a cost curve, and the
    artefact that holds such a cell -- Cuneiform at bound 6, timed on one side only -- sits
    beside the one that holds the real measurement.
    """
    module = _bo_ve(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Alpha", 6, 900, "400.0"), _o("Alpha", 7, 2700, ""))
    assert [bound for bound, _pool, _per in module.enumeration_cost()["Alpha"]] == [4, 5, 6]


# ------------------------------------------------- no word reaches the canvas without a label

# The matplotlib calls that put words on the picture. A string handed to one of these is a
# string the reader sees, so it has to come from the label file rather than from the source.
VISIBLE = {"annotate", "set_title", "set_xlabel", "set_ylabel", "suptitle", "text",
           "set_label", "set_xticklabels", "set_yticklabels"}


def _visible_literals(than: str) -> list[str]:
    import ast
    found = []
    for node in ast.walk(ast.parse(than)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr in VISIBLE and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.append(f"{node.func.attr}({first.value!r})")
        for kw in node.keywords:
            if kw.arg == "label" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                found.append(f"{node.func.attr}(label={kw.value.value!r})")
    return found


@needs_manuscript("figure_labels.vi.json")
def test_no_figure_prints_a_string_typed_into_the_source() -> None:
    """SUA 2026-09-12: the label-file checks ask the wrong question by one step.

    One asks whether a builder calls labels() at all, the other compares the file's keys with
    the keys asked for. The basin figure passed both and still printed ACO first in English on
    a Vietnamese page, for as long as the figure existed: it asked for five labels and typed
    the sixth string in place, so there was no absent key to notice and no missing call.

    Measured on the draft of 2026-09-11, page 31. The question that finds it is neither of
    those: does any word reach the canvas without passing through the label file?

    An f-string is allowed. Those carry data -- a collection name, a measured ratio -- and
    data is not language.
    """
    builders = _builders()
    assert len(builders) >= 4, f"only {len(builders)} builder(s) found; the parse is wrong"
    typed = {name: _visible_literals(than) for name, than in builders.items()}
    typed = {name: found for name, found in typed.items() if found}
    assert not typed, (
        "a figure prints words that never passed through paper/figure_labels.vi.json, so they "
        "render in whatever language the source was written in:\n  "
        + "\n  ".join(f"{name}: {', '.join(found)}" for name, found in sorted(typed.items())))


@needs_manuscript("figure_labels.vi.json")
def test_that_check_can_actually_fail() -> None:
    """The scan above finds nothing today, and a scan that cannot fire proves nothing."""
    assert _visible_literals('ax.annotate("ACO first", xy=(1, 1))') == [
        "annotate('ACO first')"]
    assert _visible_literals('ax.plot(xs, ys, label="pool size")') == [
        "plot(label='pool size')"]
    assert _visible_literals('ax.set_xlabel(text["xlabel"])') == []
    assert _visible_literals('ax.annotate(f"{name}: {ratio:.2f}", xy=(1, 1))') == []
