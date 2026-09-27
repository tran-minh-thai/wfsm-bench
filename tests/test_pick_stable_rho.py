"""The calibration report must say WHY a collection has no rho, not only that it has none.

A collection swept without a stable value and one never swept used to print the same "none"
and be counted together, and that merged label was copied into the project notes as a result
about a collection nobody had calibrated. The per-collection lookup the runners call must not
change: it prints a value or nothing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = "dataset,graph_limit,rho,stable\n"


@pytest.fixture
def tool(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pick_stable_rho", ROOT / "scripts" / "pick_stable_rho.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results, data = tmp_path / "results", tmp_path / "data"
    results.mkdir(); data.mkdir()
    for name in ("Stable", "Unstable", "Never"):
        (data / f"{name}.json").write_text("{}")
    (results / "degeneracy_rho_stable.csv").write_text(
        HEADER + "Stable,60,0.9,0\nStable,60,0.5,1\nStable,60,0.3,1\n")
    (results / "degeneracy_rho_unstable.csv").write_text(
        HEADER + "Unstable,60,0.9,0\nUnstable,60,0.5,0\n")
    monkeypatch.setattr(module, "RESULTS", results)
    monkeypatch.setattr(module, "DATA", data)
    return module


def _line(out: str, name: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(name))


def test_each_kind_of_uncalibrated_collection_gets_its_own_label(tool, capsys) -> None:
    assert tool.report() == 0
    out = capsys.readouterr().out
    assert _line(out, "Stable").split()[-1] == "0.5", "the largest stable value is reported"
    assert _line(out, "Unstable").endswith("no stable rho")
    assert _line(out, "Never").endswith("never swept")
    assert "1/3 collections calibrated" in out
    assert "1 swept without a stable rho (Unstable)" in out
    assert "1 never swept (Never)" in out


def test_the_lookup_the_runners_read_is_unchanged(tool, capsys) -> None:
    for name, expected in (("Stable", "0.5"), ("Unstable", ""), ("Never", "")):
        assert tool.for_dataset(name) == 0
        assert capsys.readouterr().out.strip() == expected, name
