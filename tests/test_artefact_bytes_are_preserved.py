"""Every results directory must keep the bytes the writing machine produced.

Line endings are evidence. A development box once overwrote an experiment-machine sweep and
left its name on the file; the only thing that still told the two apart was CRLF versus LF.
So .gitattributes marks artefacts as binary-ish, and this checks the rule covers every
results directory rather than the one that happened to exist when it was written --
results-invariant/ was created later, slipped past a rule that named results/ explicitly, and
had its bytes normalised on the very commit that created it.

The check asks git what attribute a real tracked file actually carries. Reading
.gitattributes and matching the glob by hand would re-implement git's precedence rules and
agree with itself while disagreeing with git.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked(directory: Path) -> list[str]:
    done = subprocess.run(["git", "ls-files", "--", directory.name],
                          cwd=ROOT, capture_output=True, text=True)
    return [line for line in done.stdout.splitlines() if line.strip()]


def _results_directories() -> list[Path]:
    return sorted(p for p in ROOT.glob("results*") if p.is_dir())


def test_there_is_at_least_one_results_directory_to_check() -> None:
    """Otherwise the test below passes over nothing, which is not the same as passing."""
    assert _results_directories(), "no results directory found; the check below is vacuous"


@pytest.mark.parametrize("directory", _results_directories(), ids=lambda p: p.name)
def test_git_leaves_artefact_bytes_alone(directory: Path) -> None:
    files = _tracked(directory)
    if not files:
        pytest.skip(f"{directory.name} has no tracked file yet")
    done = subprocess.run(["git", "check-attr", "text", "--", *files],
                          cwd=ROOT, capture_output=True, text=True)
    offenders = [line for line in done.stdout.splitlines() if not line.endswith(": unset")]
    assert not offenders, (
        f"{len(offenders)} of {len(files)} tracked file(s) under {directory.name}/ would have "
        f"their line endings rewritten by git; add a '-text' rule covering this directory.\n  "
        + "\n  ".join(offenders[:5]))
    print(f"\n[bytes] {directory.name}: {len(files)} tracked file(s) left byte-for-byte")
