"""Every commit recorded beside an artefact must be a commit anyone can check out.

The batch writes one environment file per stage, each carrying the commit the stage ran
under. That field is the whole point of the provenance discipline -- a reader reproduces a
number by checking out the hash beside it -- so a hash naming nothing is worse than no hash
at all: without one people ask, with a dead one they stop asking.

The parsing matters as much as the check. These files are written on the measurement machine
and `.gitattributes` keeps their CRLF endings deliberately, as evidence of which machine
produced them. A first version of this audit split on whitespace without stripping the
carriage return, asked git about `f69e028\\r`, and reported that **every** commit was
unreachable -- including ones created minutes earlier. A check that fails uniformly is
usually broken rather than alarming, and this one was.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "results").glob("environment_stage*.txt"))
PATTERN = re.compile(r"^commit\s*=\s*(\S+)", re.MULTILINE)

# Records written before the history was compressed on 2026-08-25. Their commits were real
# and their numbers are real, but the commits they name no longer exist in this repository:
# the whole history was replaced by a single commit to clean up its commit
# messages, and any rewrite invalidates every hash recorded before it.
#
# This is a LOSS, and it is declared rather than skipped. What is excused is decided by a
# PROPERTY: the record's bytes are still exactly those it had in the compression commit, the
# root of this history. A record written or rewritten since then -- a new stage, or an old
# stage run again -- is no longer excused and must name a commit that resolves; a new
# dangling hash therefore still fails.
#
# CORRECTED 2026-09-25. This used to be a list of thirteen file names, and it broke the
# re-run it should have welcomed: the first stage re-run from the current commit made its
# record resolvable, the list then refused it ("take it off the list"), and every later
# re-run session would have stopped at the test step the same way. The property needs no
# upkeep: re-running a stage changes its record, which is what takes it off.


def roots() -> list[str]:
    done = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True)
    return done.stdout.split() if done.returncode == 0 else []


def unchanged_since_compression(path: Path) -> bool:
    """True when the file's bytes equal its blob in the root commit of this history."""
    relative = path.relative_to(ROOT).as_posix()
    current = subprocess.run(["git", "hash-object", "--no-filters", str(path)], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    for root in roots():
        then = subprocess.run(["git", "rev-parse", f"{root}:{relative}"], cwd=ROOT,
                              capture_output=True, text=True)
        if then.returncode == 0 and then.stdout.strip() == current:
            return True
    return False


def recorded_commit(path: Path) -> str | None:
    # newline="" plus an explicit strip: the CR is real content here, not a parsing detail.
    found = PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    return found.group(1).strip() if found else None


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


@pytest.mark.skipif(not FILES, reason="no environment files present")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_the_commit_beside_an_artefact_exists_and_is_on_this_branch(path: Path) -> None:
    commit = recorded_commit(path)
    if commit is None:
        pytest.skip(f"{path.name} records no commit field")
    if unchanged_since_compression(path) and not git_ok("cat-file", "-e", f"{commit}^{{commit}}"):
        pytest.skip(f"{path.name} names {commit}, recorded before the 2026-08-25 compression")
    assert git_ok("cat-file", "-e", f"{commit}^{{commit}}"), (
        f"{path.name} names commit {commit!r}, which this repository does not contain. "
        f"A number produced under it cannot be reproduced by anyone."
    )
    assert git_ok("merge-base", "--is-ancestor", commit, "HEAD"), (
        f"{path.name} names commit {commit}, which exists but is not an ancestor of HEAD. "
        f"The artefact came from a branch this one does not include."
    )


@pytest.mark.skipif(not FILES, reason="no environment files present")
def test_the_audit_reaches_a_useful_number_of_files() -> None:
    """Zero parsed commits would make every check above pass by covering nothing."""
    parsed = [p for p in FILES if recorded_commit(p) is not None]
    assert len(parsed) >= 8, (
        f"only {len(parsed)} of {len(FILES)} environment files yielded a commit field; "
        f"the parse is probably broken rather than the data"
    )
