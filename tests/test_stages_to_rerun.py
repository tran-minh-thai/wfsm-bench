"""The re-run set is computed from the stage records, so the computation is what is checked.

A runner fed this list runs for a day or more. A wrong list either re-measures what was fine
or, worse, prints nothing and lets the runner fall back to its default stages.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "stages_to_rerun", ROOT / "scripts" / "stages_to_rerun.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _module()
DEAD = "deadbee"


def dead_only(commit):
    return None if commit != DEAD else "not in this repository"


def test_only_untraceable_stages_are_selected() -> None:
    order, why = S.select({16: DEAD, 19: "abc1234", 20: "abc1234"}, dead_only)
    assert order == [16], order
    assert set(why) == {16}


def test_one_calibration_stage_pulls_in_the_other_two_and_they_go_first() -> None:
    order, _ = S.select({3: DEAD, 16: DEAD, 1: "ok", 15: "ok"}, dead_only)
    assert order == [1, 3, 15, 16], order


def test_rerunning_stage_13_brings_the_determinism_check_and_puts_it_last() -> None:
    order, why = S.select({13: DEAD, 14: "ok", 16: DEAD, 18: DEAD}, dead_only)
    assert order == [13, 16, 18, 14], order
    assert "13" in why[14]


def test_a_record_without_a_commit_is_untraceable() -> None:
    assert S.untraceable_reason(None) == "record names no commit"


def test_a_windows_record_is_read_through_its_carriage_returns(tmp_path: Path) -> None:
    """The bug the first hand-written version of this check had: CRLF kept by -text."""
    path = tmp_path / "environment_stage9.txt"
    path.write_bytes(b"node = DESKTOP\r\ncommit = 9527d3f\r\ntree = clean\r\n")
    assert S.read_commit(path) == "9527d3f"


def test_a_superseded_copy_is_not_a_stage(tmp_path: Path) -> None:
    (tmp_path / "environment_stage16.txt").write_text("commit = a\n")
    (tmp_path / "environment_stage16_n267_superseded.txt").write_text("commit = b\n")
    assert S.records(tmp_path) == {16: "a"}


def test_nothing_to_rerun_prints_nothing_and_says_so(tmp_path: Path, capsys) -> None:
    """Empty stdout with exit 0 would make the runner fall back to its default stages."""
    (tmp_path / "environment_stage9.txt").write_text("commit = abc1234\n")
    code = S.main(["--results", str(tmp_path)], reason=lambda commit: None)
    out = capsys.readouterr()
    assert code == S.NOTHING_TO_DO != 0
    assert out.out == ""
    assert "[nothing]" in out.err


def test_reading_no_record_is_refused_not_reported_as_clean(tmp_path: Path, capsys) -> None:
    code = S.main(["--results", str(tmp_path)], reason=lambda commit: None)
    assert code == S.REFUSED
    assert "[refused]" in capsys.readouterr().err


def test_a_selection_prints_one_line_of_stage_numbers(tmp_path: Path, capsys) -> None:
    for stage in (1, 9, 14):
        (tmp_path / f"environment_stage{stage}.txt").write_text(f"commit = {DEAD}\n")
    code = S.main(["--results", str(tmp_path)], reason=dead_only)
    assert code == 0
    assert capsys.readouterr().out == "1 3 15 9 14\n"


def test_the_calibration_group_is_the_one_the_runner_header_names() -> None:
    header = (ROOT / "run_experiments_windows.bat").read_text(encoding="utf-8")
    named = re.search(r"must all be re-run together.*?run_experiments_windows\.bat\s+([\d ]+)",
                      header, re.S)
    assert named, "the runner header no longer names the calibration group"
    assert tuple(int(n) for n in named.group(1).split()) == S.CALIBRATION


def test_every_stage_the_script_can_name_exists_in_the_runner() -> None:
    runner = (ROOT / "run_experiments_windows.bat").read_text(encoding="utf-8")
    labels = {int(n) for n in re.findall(r"^:stage(\d+)\s*$", runner, re.M)}
    assert labels, "no :stageN labels found; the parse is wrong"
    named = set(S.CALIBRATION) | set(S.CHECKS) | {s for group in S.CHECKS.values() for s in group}
    assert named <= labels, f"stages with no label in the runner: {sorted(named - labels)}"
    on_disk = set(S.records(S.RESULTS))
    assert on_disk, "no stage records in results/; this check would compare nothing"
    assert on_disk <= labels, f"records for stages the runner lacks: {sorted(on_disk - labels)}"


def test_a_stage_with_artefacts_but_no_manifest_is_selected(tmp_path: Path, capsys) -> None:
    """Its log is on disk, its manifest is not: the commit it ran under cannot be named."""
    (tmp_path / "environment_stage9.txt").write_text("commit = abc1234\n")
    (tmp_path / "s6_methods.log").write_text("ran\n")           # written by stage 7
    code = S.main(["--results", str(tmp_path)], reason=lambda commit: None)
    captured = capsys.readouterr()
    assert code == 0 and captured.out == "7\n", captured
    assert "no environment_stage<N>.txt manifest" in captured.err


def test_rerunning_a_compared_stage_brings_the_determinism_check(tmp_path: Path, capsys) -> None:
    (tmp_path / "environment_stage9.txt").write_text("commit = abc1234\n")
    (tmp_path / "qm9_methods.log").write_text("ran\n")          # written by stage 2
    S.main(["--results", str(tmp_path)], reason=lambda commit: None)
    assert capsys.readouterr().out == "2 14\n"


def test_every_stage_in_the_runner_has_outputs_to_look_for() -> None:
    """A stage with no output pattern would never be found unrecorded -- the silent case."""
    outputs = S.stage_outputs()
    assert len(outputs) >= 20, sorted(outputs)
    empty = sorted(stage for stage, patterns in outputs.items() if not patterns)
    assert not empty, f"stages whose commands write no results/ file: {empty}"
