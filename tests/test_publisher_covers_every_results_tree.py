"""The publisher must find every artefact tree, including one added after it was written.

This is the same bug twice in the same directory. .gitattributes named results/ alone, a
results-invariant/ arrived later and slipped past it, and the bytes it existed to preserve
were normalised on the commit that created the directory; that file was fixed to match by
shape. The publisher was not, and it named results/ in three places.

Measured 2026-09-08 on the experiment machine: stage 19 finished a six-hour enumeration,
wrote its artefact to results-invariant/, and the publisher classified that artefact as
modified CODE and refused. The message sent the operator to go commit code that did not
exist, and the file had to be carried across by hand -- the route the publisher exists to
remove.

So this pins the property rather than the current answer: a tree that does not exist yet must
still be picked up on the day it appears.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _publisher():
    spec = importlib.util.spec_from_file_location(
        "publish_results", ROOT / "scripts" / "publish_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_results_directory_on_disk_is_an_artefact_tree() -> None:
    on_disk = {f"{p.name}/" for p in ROOT.iterdir()
               if p.is_dir() and p.name.startswith("results")}
    assert on_disk, "no results directory in the repository root to check against"
    found = set(_publisher().artefact_trees())
    assert found == on_disk, (
        f"the publisher works on {sorted(found)} while the repository holds {sorted(on_disk)}. "
        f"A tree it does not know about has its artefacts classified as modified code and "
        f"refused, which is how a finished stage fails to publish.")


def test_a_tree_added_later_is_picked_up(tmp_path, monkeypatch) -> None:
    """The property, not today's answer: naming the trees would pass the test above forever."""
    module = _publisher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in ("results", "results-invariant", "results-probe", "src", "scripts"):
        (tmp_path / name).mkdir()
    assert module.artefact_trees() == ["results-invariant/", "results-probe/", "results/"], (
        "a results tree created after this code was written was not picked up; the discovery "
        "is by name rather than by shape")


def test_it_refuses_rather_than_publishing_nothing(tmp_path, monkeypatch) -> None:
    """An empty answer must stop the run, not sail through reporting success."""
    module = _publisher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "src").mkdir()
    try:
        module.artefact_trees()
    except SystemExit as e:
        assert "no results tree" in str(e), f"refused for the wrong reason: {e}"
    else:
        raise AssertionError("no results tree at all, and it still returned an answer")
