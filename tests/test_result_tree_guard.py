"""The guard that keeps tests out of the result trees allows the drivers' own test log, and only it.

The Windows driver redirects the suite's output into results/logs/03_tests.log, which therefore
changes during the run the guard watches. On 2026-10-01 the guard reported that file and stopped
the measurement machine before a stage could start.
"""
from __future__ import annotations

from pathlib import Path

from conftest import changed_result_files, driver_test_logs


def test_the_windows_driver_test_log_is_found() -> None:
    found = driver_test_logs()
    assert "results/logs/03_tests.log" in found, found


def test_the_redirect_target_is_read_from_the_driver_not_from_a_list(tmp_path: Path) -> None:
    (tmp_path / "drive.bat").write_text(
        'REM %PY% -m pytest -q > results\\logs\\old.log\n'
        '%PY% -m pytest -q > results\\logs\\renamed_tests.log 2>&1\n'
        '%PY% scripts\\other.py > results\\other.log 2>&1\n', encoding="utf-8")
    assert driver_test_logs(tmp_path) == {"results/logs/renamed_tests.log"}


def test_only_the_allowed_file_is_excused() -> None:
    before = {"results/logs/03_tests.log": (0, 1), "results/a.csv": (5, 1)}
    after = {"results/logs/03_tests.log": (900, 2), "results/a.csv": (5, 1),
             "results/figures/x.pdf": (10, 2)}
    assert changed_result_files(before, after, {"results/logs/03_tests.log"}) == [
        "results/figures/x.pdf"]
    assert changed_result_files(before, after, set()) == [
        "results/figures/x.pdf", "results/logs/03_tests.log"]
