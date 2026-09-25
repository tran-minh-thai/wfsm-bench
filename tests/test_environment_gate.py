"""The gate in front of every stage must refuse for the right reason and say the right fix.

It had no test at all. Its one message sent every refusal to the development machine, and for
the commonest refusal on the measurement machine -- a publish whose push failed -- that was
worse than no advice: the commits held artefacts only, the gate ran before every stage, and so
it blocked the very publish that would have pushed them.

Each case builds a checkout and a bare remote in a scratch directory and runs the real script.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = tmp_path / "work"
    for sub in ("scripts", "results", "src"):
        (repo / sub).mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "test")
    for name in ("environment.py", "publish_results.py"):
        (repo / "scripts" / name).write_text((ROOT / "scripts" / name).read_text())
    (repo / "results" / ".keep").write_text("")
    (repo / "src" / "a.py").write_text("x = 1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "init")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")
    return repo, remote


def gate(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/environment.py", "--require-current"],
                          cwd=repo, capture_output=True, text=True)


def _advance_remote(tmp_path: Path, remote: Path) -> None:
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(remote), str(other))
    git(other, "config", "user.email", "o@example.invalid")
    git(other, "config", "user.name", "other")
    (other / "src" / "b.py").write_text("y = 1\n")
    git(other, "add", "-A"); git(other, "commit", "-qm", "code moved on"); git(other, "push", "-q")


def _artefact_commit(repo: Path, stage: int) -> None:
    (repo / "results" / f"stage_{stage}.csv").write_text("a\n1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", f"Add stage {stage} artefacts")


def test_a_current_checkout_passes(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    done = gate(repo)
    assert done.returncode == 0, done.stdout
    assert "checkout current" in done.stdout


def test_behind_is_fixed_here_with_a_pull(tmp_path: Path) -> None:
    repo, remote = _setup(tmp_path)
    _advance_remote(tmp_path, remote)
    done = gate(repo)
    assert done.returncode == 1, done.stdout
    assert "git pull --ff-only" in done.stdout
    assert "publish_results" not in done.stdout
    assert "development machine" not in done.stdout, "a pull here is the whole fix"


def test_an_unpushed_artefact_commit_gets_the_publish_command(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    _artefact_commit(repo, 14)
    done = gate(repo)
    assert done.returncode == 1, done.stdout
    assert re.search(r"publish_results\.py --stage 14\b", done.stdout), done.stdout
    assert "Add stage 14 artefacts" in done.stdout, "the stranded commit was not named"
    assert "development machine" not in done.stdout, "nothing on that machine needs fixing"


def test_an_unpushed_code_commit_is_sent_to_the_development_machine(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    _artefact_commit(repo, 14)
    (repo / "src" / "a.py").write_text("x = 2\n")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "edit made on the wrong machine")
    done = gate(repo)
    assert done.returncode == 1, done.stdout
    assert "src/a.py" in done.stdout
    assert "development machine" in done.stdout
    assert "publish_results" not in done.stdout, "publishing would push the code too"


def test_following_the_advice_opens_the_gate(tmp_path: Path) -> None:
    """The printed command is run as printed, and the gate then passes: no deadlock."""
    repo, remote = _setup(tmp_path)
    _artefact_commit(repo, 14)
    refused = gate(repo)
    command = next(line.strip() for line in refused.stdout.splitlines()
                   if "publish_results.py" in line)
    pushed = subprocess.run(command, cwd=repo, shell=True, capture_output=True, text=True)
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    assert "results/stage_14.csv" in subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "main"], cwd=remote,
        capture_output=True, text=True).stdout
    assert gate(repo).returncode == 0


def test_diverged_with_only_artefacts_is_resolved_by_the_same_command(tmp_path: Path) -> None:
    """Remote moved on AND a push failed here: the publisher rebases, so one command fixes it."""
    repo, remote = _setup(tmp_path)
    _artefact_commit(repo, 9)
    _advance_remote(tmp_path, remote)
    refused = gate(repo)
    assert refused.returncode == 1, refused.stdout
    assert re.search(r"1 ahead of and 1 behind", refused.stdout), refused.stdout
    command = next(line.strip() for line in refused.stdout.splitlines()
                   if "publish_results.py" in line)
    assert command.endswith("--stage 9"), command
    pushed = subprocess.run(command, cwd=repo, shell=True, capture_output=True, text=True)
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    assert gate(repo).returncode == 0
