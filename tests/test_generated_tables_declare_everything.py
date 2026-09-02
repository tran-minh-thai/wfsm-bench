"""The generated tables carry editorial judgements, and those must not drift from the code.

Two columns in the manuscript's tables are decisions rather than measurements: which methods
count as baselines, and what role each collection plays in the argument. Both live beside the
manuscript, so a generator reads them rather than inventing them -- and both can fall out of
step with what the repository actually contains.

A baseline added to the code and never declared would be run and never named. A collection
added to data/ and never given a role would appear in the dataset table with no stated
purpose. Neither shows up on reading, so both are checked here against the code and the data
rather than against another document.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT.parent / "paper" / "table_labels.vi.json"
PROFILE = ROOT / "results-invariant" / "dataset_profile.csv"
FIGURES = ROOT / "scripts" / "make_figures.py"

pytestmark = pytest.mark.skipif(not LABELS.exists(),
                                reason=f"the label file is outside this repository: {LABELS}")


def labels() -> dict:
    return json.loads(LABELS.read_text(encoding="utf-8"))


def _compared_methods() -> list[str]:
    spec = importlib.util.spec_from_file_location("make_figures", FIGURES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.METHODS)


def test_every_method_the_code_compares_is_declared_as_a_baseline() -> None:
    declared = {entry["key"] for entry in labels()["moi-truong"]["baselines"]}
    # The enumerator and the published miner are baselines too, and neither is a search
    # method, so neither appears in the figure builder's list.
    expected = set(_compared_methods()) | {"exhaustive", "gspan"}
    assert declared == expected, (
        f"the baseline table and the code disagree.\n"
        f"  run but never named: {sorted(expected - declared)}\n"
        f"  named but not run:   {sorted(declared - expected)}")


def test_every_baseline_declares_a_role() -> None:
    baselines = labels()["moi-truong"]["baselines"]
    # Without this, an empty list of baselines reports every baseline as declared.
    assert len(baselines) >= 5, (
        f"only {len(baselines)} baseline(s) declared; the comparison table names more, so "
        f"this check is reading the wrong key")
    empty = [e["key"] for e in baselines if not e.get("role")]
    assert not empty, f"{len(empty)} of {len(baselines)} baselines state no role: {empty}"


@pytest.mark.skipif(not PROFILE.exists(), reason="run scripts/measure_datasets.py first")
def test_every_profiled_collection_has_a_declared_role() -> None:
    with PROFILE.open(encoding="utf-8", newline="") as handle:
        collections = [row["dataset"] for row in csv.DictReader(handle)]
    assert collections, "the dataset profile is empty; the check below would be vacuous"
    roles = labels()["bo-du-lieu"]["roles"]
    undeclared = sorted(set(collections) - set(roles))
    assert not undeclared, (
        f"{len(undeclared)} of {len(collections)} collection(s) appear in the dataset table "
        f"with no stated role: {undeclared}\n"
        f"  a collection in the paper with no purpose is one a reviewer will ask about")


def test_no_role_is_declared_for_a_collection_that_is_gone() -> None:
    """A role left behind after a collection is deleted describes data nobody has."""
    if not PROFILE.exists():
        pytest.skip("run scripts/measure_datasets.py first")
    with PROFILE.open(encoding="utf-8", newline="") as handle:
        collections = {row["dataset"] for row in csv.DictReader(handle)}
    stale = sorted(set(labels()["bo-du-lieu"]["roles"]) - collections)
    assert not stale, f"roles declared for collections not in data/: {stale}"
