"""A long enumeration says it is alive, and saying so changes nothing it produces.

The size-6 stage enumerates for about six hours before its log gets a second line, and the
window the operator watches shows nothing for the whole stage. On 2026-10-02 a run was stopped
by hand because it looked hung. The enumerator now reports progress through a callback, and
the stage scripts print a line from it every ten minutes.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

from wfsmbench.baselines import mine_exhaustive
from wfsmbench.config import STUDY_SIZE_REFERENCE, Params
from wfsmbench.core.signature import canonical_signature
from wfsmbench.io import load_dataset
from wfsmbench.preprocess import build_statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Cuneiform.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RWM = _module("run_weight_mode_for_heartbeat", ROOT / "scripts" / "run_weight_mode.py")


@pytest.fixture(scope="module")
def collection():
    graphs, _ = load_dataset(DATA)
    subset = graphs[:30]
    return subset, build_statistics(subset)


def params() -> Params:
    return Params(min_support=7, max_nodes=3, max_edges=3, size_reference=STUDY_SIZE_REFERENCE)


def keyed(pool) -> dict[str, tuple[int, float]]:
    return {canonical_signature(c.particle.pattern): (c.evaluation.support, c.fitness)
            for c in pool}


def test_the_callback_sees_every_expansion_and_changes_nothing(collection) -> None:
    subset, stats = collection
    plain = mine_exhaustive(subset, stats, params()).pool
    calls: list[tuple[int, int]] = []
    watched = mine_exhaustive(subset, stats, params(),
                              progress=lambda expanded, found: calls.append((expanded, found))).pool
    assert len(plain) >= 20, f"only {len(plain)} patterns: too small to test anything"
    assert keyed(watched) == keyed(plain)
    assert [c.signature for c in watched] == [c.signature for c in plain]   # same order too
    # One call per expanded pattern, and every frequent pattern is expanded exactly once.
    assert [expanded for expanded, _found in calls] == list(range(1, len(calls) + 1))
    assert calls[-1] == (len(plain), len(plain))
    assert all(found >= expanded for expanded, found in calls)


class Clock:
    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0)


def test_silent_before_the_interval_then_one_line_per_interval() -> None:
    out = io.StringIO()
    beat = RWM.Heartbeat("size 6 supp 2 [sum]", interval=600,
                         clock=Clock(0, 10, 599, 600, 700, 1199, 1250), streams=[out])
    for expanded in range(1, 7):
        beat(expanded, expanded + 5)
    lines = out.getvalue().splitlines()
    assert len(lines) == 2 and beat.lines == 2, lines
    assert lines[0].startswith("  [progress] size 6 supp 2 [sum]: 8 frequent patterns found, 3 "
                               "expanded, 5 waiting, 10 min elapsed")
    assert "11 frequent patterns found" in lines[1] and "21 min elapsed" in lines[1]


def test_the_stage_enumeration_prints_progress_to_its_log(collection, monkeypatch, capsys) -> None:
    subset, stats = collection
    monkeypatch.setattr(RWM, "HEARTBEAT_SECONDS", 0)
    monkeypatch.setattr(RWM, "_console", lambda: None)      # keep the test off the terminal
    pool = RWM.enumerate_pool(subset, stats, params(), "size 3 supp 7 [sum]")
    printed = [line for line in capsys.readouterr().out.splitlines() if "[progress]" in line]
    assert len(printed) == len(pool), (len(printed), len(pool))
    assert printed[-1].startswith(f"  [progress] size 3 supp 7 [sum]: {len(pool)} frequent")
    assert keyed(pool) == keyed(mine_exhaustive(subset, stats, params()).pool)


def test_a_short_enumeration_prints_nothing(collection, monkeypatch, capsys) -> None:
    subset, stats = collection
    monkeypatch.setattr(RWM, "_console", lambda: pytest.fail("opened the terminal"))
    RWM.enumerate_pool(subset, stats, params(), "size 3 supp 7 [sum]")
    assert "[progress]" not in capsys.readouterr().out


def test_the_stage_script_labels_each_cell_it_enumerates() -> None:
    text = (ROOT / "scripts" / "run_weight_mode.py").read_text(encoding="utf-8")
    assert 'f"size {size} supp {support} [{shared}]")' in text
