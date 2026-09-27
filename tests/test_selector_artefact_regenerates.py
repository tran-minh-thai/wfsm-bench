"""The selector's table must be regenerable from this repository, and its step must do work.

For a month neither held. A commit deleted run_selector.py's main() together with the dead
line it meant to remove, so stage 13 ran the script as a no-op: nothing printed, exit code
zero, an empty log committed as the stage's artefact. And results/selector_regret_allpairs.csv
-- the table the selector's regret is quoted from -- had no generator in the repository and
no record of which files it was computed from. Found on 2026-09-27 by comparing the stage 13
re-run with the artefacts it replaced.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "selector_regret_allpairs.csv"


def _selector():
    spec = importlib.util.spec_from_file_location(
        "run_selector", ROOT / "scripts" / "run_selector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_all_pairs_table_regenerates_byte_for_byte(tmp_path: Path) -> None:
    selector = _selector()
    inputs = sorted({p for pattern in selector.ALL_PAIRS_RUNS
                     for p in ROOT.glob(pattern)})
    assert len(inputs) >= 6, f"only {len(inputs)} input file(s) found for the table"
    out = tmp_path / "allpairs.csv"
    assert selector.main(["--all-pairs", "--out", str(out)]) == 0
    assert out.read_bytes() == TABLE.read_bytes().replace(b"\r\n", b"\n"), (
        "the regenerated all-pairs table differs from the one in results/; either its "
        "inputs changed or ALL_PAIRS_RUNS no longer names what produced it")


def test_the_stage_13_step_prints_its_verdict_and_writes(tmp_path: Path, capsys) -> None:
    """What went missing: an invocation that printed nothing and still exited zero."""
    out = tmp_path / "selector_regret.csv"
    assert _selector().main(["--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "DISJOINT SEEDS" in printed and "[done]" in printed, printed[:300]
    assert len(out.read_text().splitlines()) > 1, "the table was written with no rows"


def test_every_script_the_windows_runners_call_has_an_entry_point() -> None:
    """A script without one runs as a no-op and exits zero, which no stage can tell apart
    from success. Checked by property -- every script either runner invokes -- so a script
    added later is covered the day it is wired in."""
    called = set()
    for runner in ("run_experiments_windows.bat", "run_all_windows.bat"):
        text = (ROOT / runner).read_text(encoding="utf-8")
        live = "\n".join(line for line in text.splitlines()
                         if not line.strip().upper().startswith("REM"))
        called |= set(re.findall(r"scripts\\(\w+)\.py", live))
    assert len(called) >= 10, f"only {len(called)} script(s) found in the runners"
    missing = sorted(name for name in called
                     if "__main__" not in (ROOT / "scripts" / f"{name}.py").read_text(
                         encoding="utf-8"))
    assert not missing, f"called by a Windows runner but has no entry point: {missing}"
