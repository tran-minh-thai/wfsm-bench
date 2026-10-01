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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"
FIGS = ROOT / "results" / "figures"
DRAFT = ROOT.parent / "paper" / "ban_thao.tex"
# The manuscript pulls each figure in by its own name, without an extension, so the artefact
# filenames live in the generator's copy map rather than anywhere in the draft.
# Checking only the prose would look clean while naming nothing.
NUMBERS_TEX = ROOT.parent / "paper" / "bang" / "so_lieu.tex"

# Figures the draft must cite. The other builders serve sections not yet written, so they are
# listed here rather than required of the draft -- and listing them is what makes adding a
# third unreferenced figure a visible decision instead of an accident.


def _declared() -> list[str]:
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [name for name, _build in module.FIGURES]


def _generator_image_map() -> dict:
    """Draft-side figure name -> artefact filename, read from the generator that copies them."""
    spec = importlib.util.spec_from_file_location(
        "sync_manuscript_numbers", DRAFT.parent / "tools" / "sync_manuscript_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.IMAGES)


@pytest.mark.skipif(not DRAFT.exists(), reason="the draft is outside this repository")
def test_the_builder_declares_the_figures_the_draft_uses() -> None:
    """CORRECTED 2026-09-11: derived from the generator instead of a set of filenames kept here.

    The list here went stale the way a list of names always does. It named four figures while
    the draft used six, so two of them were outside every check in this file -- and the two it
    missed were the two added most recently, which is exactly when a hand-kept list is least
    likely to be updated.
    """
    declared = set(_declared())
    copied = set(_generator_image_map().values())
    assert copied, "the generator copies no figure; there is nothing to check"
    missing = sorted(copied - declared)
    assert not missing, (
        f"the draft uses {missing}, which no builder in this repository produces. The file in "
        f"the draft's directory is then whatever was left there, with nothing keeping it in "
        f"step with any artefact.")


def test_every_declared_figure_is_on_disk_and_not_empty(drawn_figures) -> None:
    # A builder that declares nothing would report every declared figure as present.
    declared = _declared()
    assert len(declared) >= 4, (
        f"the builder declares {len(declared)} figure(s); the manuscript carries four, so "
        f"this check would pass over almost nothing")
    absent = [name for name in declared
              if not (drawn_figures / name).exists() or (drawn_figures / name).stat().st_size < 5000]
    assert not absent, (
        f"{len(absent)} declared figure(s) missing or truncated: {sorted(absent)}\n"
        f"  run scripts/make_figures.py")


def test_no_image_in_the_figures_directory_is_orphaned() -> None:
    r"""A picture nothing builds any more predates the current results and will still be read.

    CORRECTED 2026-09-02: glob every file, not *.png. Restricting by extension makes the check blind
    to exactly the file someone adds next -- a leftover .pdf or .svg sits in the directory,
    gets picked up by \includegraphics, which resolves by extension order rather than by which
    file is current, and is read as a fresh figure. The check also asserts the directory is not
    empty, since an empty directory has no orphans and reports a pass.
    """
    declared = set(_declared())
    if not FIGS.is_dir():
        pytest.skip(f"{FIGS.as_posix()} does not exist on this machine, so nothing in it can be stale")
    files_on_disk = sorted(p for p in FIGS.iterdir() if p.is_file() and not p.name.startswith("."))
    assert len(files_on_disk) >= 4, (
        f"only {len(files_on_disk)} file(s) in {FIGS}; with none there this check finds no orphans and "
        f"says so as if it had looked")
    orphans = sorted(p.name for p in files_on_disk if p.name not in declared)
    assert not orphans, (
        f"{len(orphans)} image(s) no builder produces: {orphans}\n"
        f"  delete them, or add the builder back -- a stale picture is read as a fresh one")


def _included_figures(tex: str) -> tuple[set[str], list[str]]:
    """({figure name}, [include arguments that are not a figure in hinh/]) for one draft.

    Comments go first: a figure taken off the page with a leading % is not on the page, and
    matching the raw text counted it as included. Every include argument is then read, not only
    the ones already in the expected shape -- an argument outside that shape (an extension, a
    different directory) used to be invisible, so a stale picture pulled in as hinh/old.pdf
    passed. Here it is returned, and reported, instead.
    """
    visible = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())
    names, odd = set(), []
    for argument in re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", visible):
        shape = re.fullmatch(r"(?:\./)?hinh/([A-Za-z0-9_-]+)", argument.strip())
        if shape:
            names.add(shape.group(1))
        else:
            odd.append(argument.strip())
    return names, odd


# Every draft that exists is checked, not one named file. The English manuscript became the
# submission version on 2026-09-25 while this check still read only the Vietnamese draft, so
# the file that would be submitted was outside it.
DRAFTS = [ROOT.parent / "paper" / "manuscript.tex", DRAFT]


@pytest.mark.parametrize("draft", DRAFTS, ids=lambda p: p.name)
def test_every_drawn_figure_is_included_by_the_draft(draft: Path) -> None:
    """CORRECTED 2026-09-11: ask whether the draft INCLUDES the figure, not whether it names the file.

    CORRECTED 2026-09-25: two blind spots, both passing silently, found by feeding the check
    altered drafts. A figure commented out with % still counted as included, and an include
    written with an extension (hinh/old.pdf) was not seen at all, so a stale picture on the page
    passed. Of six altered drafts the old version caught three; see _included_figures.

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
    if not draft.exists():
        pytest.skip(f"{draft.name} is outside this repository and not present here")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sync_manuscript_numbers", draft.parent / "tools" / "sync_manuscript_numbers.py")
    generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator_module)

    included, odd = _included_figures(draft.read_text(encoding="utf-8"))
    assert included, f"{draft.name} includes no figure at all; the pattern stopped matching"
    assert not odd, (
        f"{draft.name} includes {odd}, which is not a figure by name in hinh/. The generator "
        f"copies into hinh/ without an extension in the name, so anything else on the page is "
        f"a file nothing keeps current.")

    copied_names = set(generator_module.IMAGES)
    assert copied_names, "the generator copies no figure; there is nothing to check"

    lacking = sorted(copied_names - included)
    assert not lacking, (
        f"the generator copies {lacking} into the draft's figure directory and no include line "
        f"puts {'them' if len(lacking) > 1 else 'it'} on a page; a figure nobody sees is a "
        f"figure that still has to be kept in step with its artefact")

    uninvited = sorted(included - copied_names)
    assert not uninvited, (
        f"the draft includes {uninvited}, which the generator does not copy. Those files are whatever "
        f"was last left in the directory, and nothing keeps them current with any artefact.")


def test_the_builder_runs_and_reproduces_every_figure(tmp_path: Path) -> None:
    """A builder that has not been run since its inputs changed proves nothing about them."""
    done = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path)], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, f"the figure builder failed\n{done.stdout}{done.stderr}"
    for name in _declared():
        assert (tmp_path / name).exists(), f"{name} was declared but not written"


# --------------------------------------------------------- Vietnamese labels on every figure

LABELS = ROOT / "figures" / "labels.en.json"


def _builders() -> dict[str, str]:
    """{figure filename: source of the function that draws it}, read with ast.

    By name would be guesswork: the builders are not named after their outputs.
    """
    import ast
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    body_text = {node.name: ast.get_source_segment(text, node)
            for node in tree.body if isinstance(node, ast.FunctionDef)}
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: body_text[build.__name__] for name, build in module.FIGURES}


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
    silent = sorted(name for name, body_text in builders.items() if "labels(" not in body_text)
    assert not silent, (
        f"{len(silent)} of {len(builders)} figure(s) never ask the label file for anything, so "
        f"they render in whatever language their source was written in: {silent}")


def test_no_label_key_is_orphaned_and_none_is_missing() -> None:
    """Both directions. An unused key is a figure someone deleted; a missing one is English."""
    import json
    import re
    keys = {k for k in json.loads(LABELS.read_text(encoding="utf-8")) if not k.startswith("_")}
    asked = set()
    for body_text in _builders().values():
        asked |= set(re.findall(r'labels\(\s*["\']([^"\']+)["\']\s*\)', body_text))
    assert asked, "no builder asks for a label key; the parse is wrong"
    lacking = sorted(asked - keys)
    unused_keys = sorted(keys - asked)
    assert not lacking, f"builders ask for keys the label file does not have: {lacking}"
    assert not unused_keys, (
        f"the label file carries keys no builder asks for: {unused_keys}. Either a figure was "
        f"deleted and its labels stayed, or a builder stopped asking.")


# ------------------------------------------- the cost figure refuses a curve through two dots

HEADER_ROW = ("dataset,graph_limit,size_bound,min_support,gspan_patterns,gspan_seconds,"
       "exhaustive_patterns,exhaustive_seconds,speedup,sets_agree,supports_disagreeing,"
       "gspan_optimum_edges,gspan_optimum_fitness")


def _o(dataset: str, bound: int, pool: int, seconds: str) -> str:
    """One cell of the timed enumeration artefact. Only four fields are read here."""
    return f"{dataset},200,{bound},2,{pool},1.000,{pool},{seconds},1.0,1,0,3,0.1"


def _builder_on(tmp_path: Path, *rows: str):
    """A module bound to a results directory holding exactly the cells named."""
    (tmp_path / "gspan_baseline_probe.csv").write_text(
        "\n".join((HEADER_ROW, *rows)) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("make_figures_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULTS = tmp_path
    return module


def test_the_cost_figure_refuses_a_collection_measured_at_two_bounds(tmp_path: Path) -> None:
    """Two points join into a straight line whatever the truth is.

    This is the refusal the figure exists behind: it could have been drawn at any time from
    the two bounds already on disk, and it would have drawn a growth rate nothing measured.
    The third bound cost nine hours on the measurement machine, so the refusal has to be
    demonstrated rather than asserted -- a guard nobody has seen fail is a guard nobody knows
    is wired up.
    """
    module = _builder_on(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Beta", 4, 100, "10.0"), _o("Beta", 5, 300, "60.0"),
                    _o("Beta", 6, 900, "400.0"))
    with pytest.raises(SystemExit) as refused:
        module.figure_enumeration_cost(tmp_path / "enumeration_cost_by_bound.pdf")
    assert "Alpha (2)" in str(refused.value), str(refused.value)
    assert "Beta" not in str(refused.value), "the collection with three bounds is not at fault"
    assert not (tmp_path / "enumeration_cost_by_bound.pdf").exists(), "it refused and drew anyway"


def test_the_cost_figure_draws_once_every_collection_has_three(tmp_path: Path) -> None:
    """The other half of the refusal: it must pass on data that earns it."""
    module = _builder_on(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Alpha", 6, 900, "400.0"))
    module.figure_enumeration_cost(tmp_path / "enumeration_cost_by_bound.pdf")
    assert (tmp_path / "enumeration_cost_by_bound.pdf").stat().st_size > 5000


def test_the_cost_figure_refuses_a_cell_two_artefacts_disagree_on(tmp_path: Path) -> None:
    """One cell, two measurements: the figure cannot pick, so it must not pick silently."""
    module = _builder_on(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Alpha", 6, 900, "400.0"), _o("Alpha", 6, 900, "700.0"))
    with pytest.raises(SystemExit) as refused:
        module.figure_enumeration_cost(tmp_path / "enumeration_cost_by_bound.pdf")
    assert "measured twice" in str(refused.value), str(refused.value)


def test_the_cost_figure_ignores_a_cell_with_no_enumeration_time(tmp_path: Path) -> None:
    """A gSpan-only cell carries a pool size and no seconds.

    Dividing by a time nobody measured would put a made-up point on a cost curve, and the
    artefact that holds such a cell -- Cuneiform at bound 6, timed on one side only -- sits
    beside the one that holds the real measurement.
    """
    module = _builder_on(tmp_path, _o("Alpha", 4, 100, "10.0"), _o("Alpha", 5, 300, "60.0"),
                    _o("Alpha", 6, 900, "400.0"), _o("Alpha", 7, 2700, ""))
    assert [bound for bound, _pool, _per in module.enumeration_cost()["Alpha"]] == [4, 5, 6]


# ------------------------------------------------- no word reaches the canvas without a label

# The matplotlib calls that put words on the picture. A string handed to one of these is a
# string the reader sees, so it has to come from the label file rather than from the source.
VISIBLE = {"annotate", "set_title", "set_xlabel", "set_ylabel", "suptitle", "text",
           "set_label", "set_xticklabels", "set_yticklabels"}


def _visible_literals(body_text: str) -> list[str]:
    import ast
    found = []
    for node in ast.walk(ast.parse(body_text)):
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


def test_no_figure_prints_a_string_typed_into_the_source() -> None:
    """CORRECTED 2026-09-12: the label-file checks ask the wrong question by one step.

    One asks whether a builder calls labels() at all, the other compares the file's keys with
    the keys asked for. The basin figure passed both and still printed ACO first in English on
    a Vietnamese page, for as long as the figure existed: it asked for five labels and typed
    the sixth string in place, so there was no absent key to notice and no missing call.

    Measured on a draft of 2026-09-11. The question that finds it is neither of
    those: does any word reach the canvas without passing through the label file?

    An f-string is allowed. Those carry data -- a collection name, a measured ratio -- and
    data is not language.
    """
    builders = _builders()
    assert len(builders) >= 4, f"only {len(builders)} builder(s) found; the parse is wrong"
    typed = {name: _visible_literals(body_text) for name, body_text in builders.items()}
    typed = {name: found for name, found in typed.items() if found}
    assert not typed, (
        "a figure prints words that never passed through figures/labels.en.json, so they "
        "render in whatever language the source was written in:\n  "
        + "\n  ".join(f"{name}: {', '.join(found)}" for name, found in sorted(typed.items())))


def test_that_check_can_actually_fail() -> None:
    """The scan above finds nothing today, and a scan that cannot fire proves nothing."""
    assert _visible_literals('ax.annotate("ACO first", xy=(1, 1))') == [
        "annotate('ACO first')"]
    assert _visible_literals('ax.plot(xs, ys, label="pool size")') == [
        "plot(label='pool size')"]
    assert _visible_literals('ax.set_xlabel(text["xlabel"])') == []
    assert _visible_literals('ax.annotate(f"{name}: {ratio:.2f}", xy=(1, 1))') == []
