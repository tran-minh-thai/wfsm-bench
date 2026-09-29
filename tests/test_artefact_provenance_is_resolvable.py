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
# the whole history was replaced by a single commit to clean up its commit messages, and
# any rewrite invalidates every hash recorded before it.
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


def roots(repo: Path = ROOT) -> list[str]:
    done = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo,
                          capture_output=True, text=True)
    return done.stdout.split() if done.returncode == 0 else []


def unchanged_since_compression(path: Path, repo: Path = ROOT) -> bool:
    """True when the file's bytes equal its blob in the root commit of this history."""
    relative = path.relative_to(repo).as_posix()
    current = subprocess.run(["git", "hash-object", "--no-filters", str(path)], cwd=repo,
                             capture_output=True, text=True).stdout.strip()
    for root in roots(repo):
        then = subprocess.run(["git", "rev-parse", f"{root}:{relative}"], cwd=repo,
                              capture_output=True, text=True)
        if then.returncode == 0 and then.stdout.strip() == current:
            return True
    return False


def recorded_commit(path: Path) -> str | None:
    """The commit a record names, translated through COMMIT_MAP.tsv when it predates the
    2026-09-29 rewrite of main (scripts/commit_map.py)."""
    # newline="" plus an explicit strip: the CR is real content here, not a parsing detail.
    found = PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    if not found:
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("commit_map", ROOT / "scripts" / "commit_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(found.group(1).strip())


def git_ok(*args: str, repo: Path = ROOT) -> bool:
    return subprocess.run(["git", *args], cwd=repo,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def excused(path: Path, commit: str, repo: Path = ROOT) -> bool:
    """A pre-compression record: bytes unchanged since the root, commit outside the history.

    CORRECTED 2026-09-25, the same day the property replaced the list. Its first version
    asked whether the commit EXISTS ("cat-file -e") instead of whether it is REACHABLE. On
    the development machine the old commits are gone, so the two agree; the measurement
    machine's clone predates the compression and still holds them as unreachable objects,
    so there the excuse never applied and all twelve old records failed. Reachability is
    the property that matters: an object only one machine still happens to hold lets
    nobody reproduce a number from a clone.
    """
    return (unchanged_since_compression(path, repo)
            and not git_ok("merge-base", "--is-ancestor", commit, "HEAD", repo=repo))


@pytest.mark.skipif(not FILES, reason="no environment files present")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_the_commit_beside_an_artefact_exists_and_is_on_this_branch(path: Path) -> None:
    commit = recorded_commit(path)
    if commit is None:
        pytest.skip(f"{path.name} records no commit field")
    if excused(path, commit):
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


def test_the_excuse_does_not_depend_on_which_old_objects_a_machine_still_holds(
        tmp_path: Path) -> None:
    """Every state a record can be in, built in a scratch repository.

    The case that broke on the measurement machine is the second: a pre-compression commit
    that still EXISTS in that clone's object store but is on no branch. It cannot occur on a
    machine that cloned after the compression, which is why a check run only there missed it.
    """
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)

    def run(*args: str, stdin: str = "") -> str:
        return subprocess.run(["git", *args], cwd=repo, input=stdin, capture_output=True,
                              text=True, check=True).stdout.strip()

    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "test")
    # As in the real repository: artefact bytes are stored as written. Without it a machine
    # with core.autocrlf=input stores these CRLF records as LF, and "unchanged since the
    # root" compares CRLF on disk with LF in the blob -- which is how this test first failed.
    (repo / ".gitattributes").write_text("results*/** -text\n")
    empty_tree = run("mktree")
    orphan = run("commit-tree", empty_tree, "-m", "made before the compression")[:7]
    assert git_ok("cat-file", "-e", f"{orphan}^{{commit}}", repo=repo), "orphan not created"

    kept = repo / "results" / "environment_stage1.txt"      # orphan that still exists
    gone = repo / "results" / "environment_stage2.txt"      # commit absent altogether
    kept.write_bytes(f"node = X\r\ncommit = {orphan}\r\n".encode())
    gone.write_bytes(b"node = X\r\ncommit = deadbee\r\n")
    run("add", "-A"); run("commit", "-qm", "the compression")

    assert excused(kept, orphan, repo), "an unreachable orphan still in the store is excused"
    assert excused(gone, "deadbee", repo), "an absent commit is excused"

    # A stage re-run from the current commit: its record changes and names a reachable
    # commit, so it is no longer excused -- and the strict check it then faces passes.
    head = run("rev-parse", "--short", "HEAD")
    kept.write_bytes(f"node = X\r\ncommit = {head}\r\n".encode())
    assert not excused(kept, head, repo)
    assert git_ok("merge-base", "--is-ancestor", head, "HEAD", repo=repo)

    # A record rewritten with a dead commit is not excused either, so the strict check fails.
    gone.write_bytes(b"node = Y\r\ncommit = deadbee\r\n")
    assert not excused(gone, "deadbee", repo)



def test_a_rewritten_commit_resolves_through_the_map(tmp_path: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("commit_map", ROOT / "scripts" / "commit_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = tmp_path / "COMMIT_MAP.tsv"
    table.write_text("# old\tnew\tsubject\n" + "a" * 40 + "\t" + "b" * 40 + "\tone\n"
                     + "a" * 7 + "c" * 33 + "\t" + "d" * 40 + "\ttwo\n")
    assert module.resolve("a" * 8, table) == "b" * 40
    assert module.resolve("1234567", table) == "1234567"          # not listed: unchanged
    with pytest.raises(ValueError):
        module.resolve("a" * 7, table)                             # ambiguous: refused
    mapped = module.load()
    assert len(mapped) >= 70, f"COMMIT_MAP.tsv lists {len(mapped)} commits"
    for new in list(mapped.values())[:: max(1, len(mapped) // 10)]:
        assert git_ok("merge-base", "--is-ancestor", new, "HEAD"), f"{new} is not on this branch"
