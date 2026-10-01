"""Every script a stage runs keeps a Windows machine awake, without changing what it prints.

The Windows call cannot be made on the development machine, so it is checked against a stand-in
that records what it was asked; the real call is first exercised by the measurement machine,
where the script prints "[awake]" on stderr at the start of every stage log.
"""
from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AWAKE = _module("keep_awake", ROOT / "scripts" / "keep_awake.py")


class Kernel32:
    def __init__(self, answer: int) -> None:
        self.answer, self.calls = answer, []

    def SetThreadExecutionState(self, flags: int) -> int:     # noqa: N802 -- the Windows name
        self.calls.append(flags)
        return self.answer


def test_on_windows_it_asks_to_stay_awake_for_the_life_of_the_process() -> None:
    kernel, err = Kernel32(answer=0x80000000), io.StringIO()
    assert AWAKE.keep_awake(kernel32=kernel, platform="win32", stream=err) is True
    assert kernel.calls == [0x80000000 | 0x00000001]          # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    assert "[awake]" in err.getvalue()


def test_a_refused_request_is_reported_not_ignored() -> None:
    kernel, err = Kernel32(answer=0), io.StringIO()
    assert AWAKE.keep_awake(kernel32=kernel, platform="win32", stream=err) is False
    assert "[warning]" in err.getvalue()


def test_elsewhere_it_does_nothing() -> None:
    kernel, err = Kernel32(answer=1), io.StringIO()
    assert AWAKE.keep_awake(kernel32=kernel, platform="darwin", stream=err) is False
    assert kernel.calls == [] and err.getvalue() == ""


def test_every_script_a_stage_runs_keeps_the_machine_awake() -> None:
    commands = _module("stage_commands", ROOT / "scripts" / "stage_commands.py").stage_commands()
    scripts = sorted({name for lines in commands.values() for line in lines
                      for name in re.findall(r"scripts/(\w+\.py)", line)})
    assert len(scripts) >= 8, f"only {len(scripts)} scripts found in the stage commands: {scripts}"
    missing = []
    for name in scripts:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        entry = text.split('if __name__ == "__main__":', 1)
        if len(entry) != 2 or "keep_awake()" not in entry[1]:
            missing.append(name)
    assert not missing, f"stage scripts that let Windows sleep: {missing}"


@pytest.mark.parametrize("dataset,expects_output", [("QM9_bond", True), ("DHFR_bond", False)])
def test_simulated_windows_leaves_the_stdout_the_driver_reads_unchanged(dataset, expects_output) -> None:
    """The driver picks the weight modes from whether pick_stable_rho.py prints anything, so
    the case that matters most is a dataset with no stable rho: one stray line flips it."""
    plain = subprocess.run([sys.executable, "scripts/pick_stable_rho.py", "--dataset", dataset],
                           cwd=ROOT, capture_output=True, text=True)
    simulated = subprocess.run([sys.executable, "-c", (
        "import runpy, sys; sys.path.insert(0, 'scripts'); import keep_awake\n"
        "class K:\n"
        "    def SetThreadExecutionState(self, flags): return 1\n"
        "keep_awake.keep_awake.__defaults__ = (K(), 'win32', None)\n"
        f"sys.argv = ['scripts/pick_stable_rho.py', '--dataset', '{dataset}']\n"
        "runpy.run_path('scripts/pick_stable_rho.py', run_name='__main__')\n")],
        cwd=ROOT, capture_output=True, text=True)
    assert "[awake]" in simulated.stderr, simulated.stderr     # the simulation reached the call
    assert simulated.stdout == plain.stdout
    assert bool(plain.stdout.strip()) is expects_output, plain.stdout
