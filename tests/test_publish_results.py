"""publish_results decides what may leave the measurement machine, so its refusals matter.

Three behaviours are worth holding: it refuses when code is modified, it names files git is
ignoring instead of losing them, and it reports nothing-to-do rather than succeeding
silently. Each is checked in a scratch repository so the real one is never touched.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_results.py"


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=cwd, capture_output=True, text=True)


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch) -> Path:
    """A throwaway repository with the script pointed at it via its parents[1] anchor."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "results").mkdir()
    (repo / "src").mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "test")
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "results" / ".keep").write_text("")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    (repo / "scripts" / "publish_results.py").write_text(SCRIPT.read_text())
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "script")
    return repo


def bare_remote(tmp_path: Path) -> Path:
    """A bare remote whose HEAD names main.

    The default branch is a per-machine git setting: main on the development machine, master
    on the measurement machine. A bare remote whose HEAD names a branch nobody pushes clones
    EMPTY, so a test that clones it to play "someone else pushed" pushed to a new master
    instead -- main never moved, and the case under test never happened. Measured there on
    2026-09-25: two tests failed outright, and the merge-commit test passed without testing.
    """
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    return remote


def remote_main_count(remote: Path) -> int:
    return int(subprocess.run(["git", "rev-list", "--count", "main"], cwd=remote,
                              capture_output=True, text=True, check=True).stdout)


def publish(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/publish_results.py", *args],
                          cwd=repo, capture_output=True, text=True)


def test_refuses_when_code_is_modified(scratch: Path) -> None:
    (scratch / "src" / "a.py").write_text("x = 2\n")
    done = publish(scratch, "--stage", "9", "--dry-run")
    assert done.returncode == 1, done.stdout
    assert "modified outside results/" in done.stderr
    assert "src/a.py" in done.stderr


def test_names_the_files_git_is_ignoring(scratch: Path) -> None:
    (scratch / ".gitignore").write_text("results/*.csv\n")
    git(scratch, "add", "-A")
    git(scratch, "commit", "-qm", "ignore")
    (scratch / "results" / "measured.csv").write_text("a,b\n1,2\n")
    (scratch / "results" / "measured.log").write_text("ok\n")
    done = publish(scratch, "--stage", "9", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "results/measured.csv" in done.stdout, "the ignored artefact was not named"
    assert "[WARNING]" in done.stdout, "a file written by this run must warn loudly"


def test_a_long_standing_exclusion_is_counted_not_shouted_about(scratch: Path) -> None:
    """Warning about every deliberate exclusion every time makes the warning unreadable."""
    (scratch / "results" / "old.csv").write_text("a\n1\n")
    (scratch / ".gitignore").write_text("results/*.csv\n")
    git(scratch, "add", "-A")
    git(scratch, "commit", "-qm", "ignore")
    # Age the file so it predates the commit, as a long-standing exclusion does.
    import os, time
    stamp = time.time() - 3600
    os.utime(scratch / "results" / "old.csv", (stamp, stamp))
    done = publish(scratch, "--stage", "9", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "[WARNING]" not in done.stdout, "an untouched exclusion must not shout"
    assert "long-standing exclusion" in done.stdout


def test_reports_nothing_to_do_rather_than_succeeding_quietly(scratch: Path) -> None:
    done = publish(scratch, "--stage", "9", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "[nothing]" in done.stdout


def test_stages_only_results(scratch: Path) -> None:
    (scratch / "results" / "new.log").write_text("measured\n")
    done = publish(scratch, "--stage", "9", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "results/new.log" in done.stdout
    assert "[dry-run]" in done.stdout


def test_it_actually_commits_and_pushes(tmp_path: Path) -> None:
    """The happy path, end to end against a real remote.

    Everything above checks refusals. Telling someone to run a command that pushes, while
    only its refusals are tested, leaves the one path that changes a remote unverified.
    """
    remote = bare_remote(tmp_path)

    repo = tmp_path / "work"
    (repo / "scripts").mkdir(parents=True)
    (repo / "results").mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "test")
    (repo / "scripts" / "publish_results.py").write_text(SCRIPT.read_text())
    (repo / "results" / ".keep").write_text("")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")

    # What a stage leaves behind.
    (repo / "results" / "pareto_size_BZR_bond.csv").write_text("dataset,pool\nBZR_bond,359\n")
    (repo / "results" / "pareto_size_BZR_bond.log").write_text("[mine] 359 patterns\n")

    done = subprocess.run([sys.executable, "scripts/publish_results.py", "--stage", "17"],
                          cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "[ok] pushed" in done.stdout, done.stdout

    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", "main"], cwd=remote,
                            capture_output=True, text=True).stdout
    assert "results/pareto_size_BZR_bond.csv" in listed, "the CSV never reached the remote"
    assert "results/pareto_size_BZR_bond.log" in listed, "the log never reached the remote"

    # And a second call with nothing new must not create an empty commit.
    before = subprocess.run(["git", "rev-list", "--count", "main"], cwd=remote,
                            capture_output=True, text=True).stdout.strip()
    again = subprocess.run([sys.executable, "scripts/publish_results.py", "--stage", "17"],
                           cwd=repo, capture_output=True, text=True)
    assert again.returncode == 0 and "[nothing]" in again.stdout, again.stdout
    after = subprocess.run(["git", "rev-list", "--count", "main"], cwd=remote,
                           capture_output=True, text=True).stdout.strip()
    assert before == after, "a second call with nothing new still moved the remote"


def test_it_does_not_carry_a_merge_commit(tmp_path: Path) -> None:
    """Rebase, not merge: the branch's own instructions forbid a merge made here."""
    remote = bare_remote(tmp_path)
    repo = tmp_path / "work"
    (repo / "scripts").mkdir(parents=True)
    (repo / "results").mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "test")
    (repo / "scripts" / "publish_results.py").write_text(SCRIPT.read_text())
    (repo / "results" / ".keep").write_text("")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "init")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")

    # Someone else advances the branch while the stage was running.
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(remote), str(other))
    git(other, "config", "user.email", "o@example.invalid")
    git(other, "config", "user.name", "other")
    before = remote_main_count(remote)
    (other / "note.txt").write_text("elsewhere\n")
    git(other, "add", "-A"); git(other, "commit", "-qm", "elsewhere"); git(other, "push", "-q")
    assert remote_main_count(remote) == before + 1, "main did not advance; nothing to merge"

    (repo / "results" / "measured.log").write_text("done\n")
    done = subprocess.run([sys.executable, "scripts/publish_results.py", "--stage", "17"],
                          cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    merges = subprocess.run(["git", "log", "--merges", "--oneline", "main"], cwd=remote,
                            capture_output=True, text=True).stdout.strip()
    assert merges == "", f"a merge commit reached the remote:\n{merges}"


# ------------------------------------------------- a commit whose push failed must still leave

def _pushed_repo(tmp_path: Path) -> tuple[Path, Path]:
    """(work, remote): a checkout already in step with a bare remote, as after a good stage."""
    remote = bare_remote(tmp_path)
    repo = tmp_path / "work"
    (repo / "scripts").mkdir(parents=True)
    (repo / "results").mkdir()
    (repo / "src").mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "test")
    (repo / "scripts" / "publish_results.py").write_text(SCRIPT.read_text())
    (repo / "results" / ".keep").write_text("")
    (repo / "src" / "a.py").write_text("x = 1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "init")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")
    return repo, remote


def _remote_files(remote: Path) -> str:
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", "main"], cwd=remote,
                          capture_output=True, text=True).stdout


def test_a_commit_whose_push_failed_is_pushed_by_the_next_call(tmp_path: Path) -> None:
    """The state a failed push leaves: artefacts committed here, nothing new to stage.

    Before 2026-09-25 the next call printed [nothing] and exited zero, and the commit stayed
    on the measurement machine while every later call reported all was well.
    """
    repo, remote = _pushed_repo(tmp_path)
    (repo / "results" / "stage_18.csv").write_text("dataset,pool\nBZR_bond,359\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "Add stage 18 artefacts")
    assert "results/stage_18.csv" not in _remote_files(remote), "the setup already pushed"

    done = publish(repo, "--stage", "19")
    assert done.returncode == 0, done.stderr
    assert "[unpushed]" in done.stdout and "[nothing]" not in done.stdout, done.stdout
    assert "[ok] pushed 1 earlier commit(s)" in done.stdout, done.stdout
    assert "results/stage_18.csv" in _remote_files(remote), "the committed artefact never left"

    again = publish(repo, "--stage", "19")
    assert again.returncode == 0 and "[nothing]" in again.stdout, again.stdout


def test_a_dry_run_names_the_unpushed_commit_and_moves_nothing(tmp_path: Path) -> None:
    repo, remote = _pushed_repo(tmp_path)
    (repo / "results" / "stage_18.csv").write_text("a\n1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "Add stage 18 artefacts")
    done = publish(repo, "--stage", "19", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "Add stage 18 artefacts" in done.stdout, "the unpushed commit was not named"
    assert "[dry-run]" in done.stdout
    assert "results/stage_18.csv" not in _remote_files(remote), "a dry run moved the remote"


def test_refuses_to_carry_an_unpushed_commit_that_touches_code(tmp_path: Path) -> None:
    """Pushing earlier commits must not become a way for code to leave the measurement machine.

    Checked on both paths: with nothing new to stage, and with a fresh artefact to publish,
    since a push from either carries every earlier unpushed commit along.
    """
    repo, remote = _pushed_repo(tmp_path)
    (repo / "src" / "a.py").write_text("x = 2\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "edit made on the wrong machine")

    done = publish(repo, "--stage", "19")
    assert done.returncode == 1, done.stdout
    assert "src/a.py" in done.stderr, done.stderr

    (repo / "results" / "fresh.csv").write_text("a\n1\n")
    with_artefact = publish(repo, "--stage", "19")
    assert with_artefact.returncode == 1, with_artefact.stdout
    assert "src/a.py" in with_artefact.stderr, with_artefact.stderr
    assert "x = 1" in subprocess.run(["git", "show", "main:src/a.py"], cwd=remote,
                                     capture_output=True, text=True).stdout, \
        "the code edit reached the remote"


def test_no_origin_is_reported_as_unverified(scratch: Path) -> None:
    """Without a remote nothing can be compared, and that must not read as all pushed."""
    done = publish(scratch, "--stage", "9", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "[unverified]" in done.stdout, done.stdout
