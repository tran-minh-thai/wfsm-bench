r"""Commit and push what a stage wrote, and refuse the ways this has gone wrong before.

Every round so far ended with the artefacts pushed by hand, and both hand-pushes went wrong
in a different way. One merged instead of rebasing, putting a merge commit on a branch whose
own instructions forbid it. The other pushed the logs and left the CSVs behind, because
`.gitignore` still carried a glob from a deleted runner and swallowed them without a word --
the run printed [ok] and the artefact simply never existed for anyone else.

So this does four things, and the third is the reason it exists:

  1. refuses if anything outside the artefact trees is modified. A stage publishes
     measurements, not code; code reaches this machine by git and leaves the same way, from
     the development machine.
  2. stages the artefact trees only.
  3. **names every file under them that git is ignoring.** That is the silent loss, and
     naming it is the whole point: a file that exists on disk and cannot reach a commit
     should not be discoverable only by someone later asking where it went.
  4. rebases rather than merges, then pushes.

    python scripts/publish_results.py --stage 17
    python scripts/publish_results.py --stage 17 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise SystemExit(f"[failed] git {' '.join(args)}\n{done.stderr.strip()}")
    return done.stdout


def artefact_trees() -> list[str]:
    """Every results tree, found by the SHAPE of its name rather than a list to keep current.

    .gitattributes already learned this lesson and says so in its own comment: its first
    version named results/ alone, a results-invariant/ added later slipped straight past it,
    and the bytes it existed to preserve were normalised on the very commit that created the
    directory. This file carried the same bug far longer -- it named results/ in three places,
    so a stage writing to results-invariant/ had its output classified as MODIFIED CODE and
    was refused.

    Measured 2026-09-08 on the experiment machine: stage 19 finished a six-hour enumeration,
    wrote results-invariant/gspan_agreement_full_Cuneiform.csv, and could not publish it. The
    refusal message told the operator to commit their code on the development machine, which
    was advice about a problem that did not exist, and the artefact had to be carried across
    by hand -- the exact route this script exists to remove.
    """
    trees = sorted(f"{path.name}/" for path in ROOT.iterdir()
                   if path.is_dir() and path.name.startswith("results"))
    if not trees:
        raise SystemExit(
            "[refused] no results tree in the repository root. Publishing nothing while "
            "reporting success is how an artefact goes missing quietly.")
    return trees


def dirty_outside_results() -> list[str]:
    trees = artefact_trees()
    lines = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    return [line for line in lines
            if not any(line[3:].startswith(tree) for tree in trees)]


def ignored_under_results() -> tuple[list[str], int]:
    """(files this run wrote that git ignores, count of long-standing exclusions).

    Splitting the two matters. Nine paths under results/ are excluded on purpose -- per-seed
    output, figures, logs -- so warning about all of them every time produces a warning
    nobody reads, and the one that matters hides inside it. The dangerous case is narrow: a
    file *this run just wrote* that cannot reach a commit. Everything written since the last
    commit is this run's output, so the commit's timestamp separates them without any list
    of names to go stale.
    """
    out = git("status", "--porcelain", "--ignored", *artefact_trees())
    paths = sorted(line[3:] for line in out.splitlines() if line.startswith("!!"))
    try:
        since = int(git("log", "-1", "--format=%ct").strip())
    except (SystemExit, ValueError):
        return paths, 0
    fresh, old = [], 0
    for name in paths:
        target = ROOT / name
        newest = 0.0
        if target.is_dir():
            for child in target.rglob("*"):
                if child.is_file():
                    newest = max(newest, child.stat().st_mtime)
        elif target.exists():
            newest = target.stat().st_mtime
        if newest > since:
            fresh.append(name)
        else:
            old += 1
    return fresh, old


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, help="stage number, for the commit message")
    ap.add_argument("--dry-run", action="store_true", help="report and stop before committing")
    args = ap.parse_args()

    outside = dirty_outside_results()
    if outside:
        print(f"[refused] these are modified outside {' '.join(artefact_trees())}:",
              file=sys.stderr)
        for line in outside:
            print(f"          {line}", file=sys.stderr)
        print("          A stage publishes measurements, not code. Commit code on the\n"
              "          development machine and pull it here.", file=sys.stderr)
        return 1

    fresh_ignored, old_ignored = ignored_under_results()
    if old_ignored:
        print(f"[note] {old_ignored} long-standing exclusion(s) under the artefact trees "
              f"untouched by this run")
    if fresh_ignored:
        print(f"[WARNING] this run wrote {len(fresh_ignored)} file(s) that git ignores. They "
              f"exist on disk and cannot reach a commit:")
        for name in fresh_ignored:
            print(f"          {name}")
        print("          A run printing [ok] while its artefact never leaves the machine is "
              "how the stage 16 CSVs were lost. If any of these is an artefact, .gitignore "
              "is stale -- fix it on the development machine rather than forcing the add "
              "here.")

    git("add", "--", *artefact_trees())
    staged = [line for line in git("diff", "--cached", "--name-only").splitlines() if line]
    if not staged:
        print(f"[nothing] no change under {' '.join(artefact_trees())} to publish. "
              f"Nothing was written, or it was already committed.")
        return 0

    print(f"[staged] {len(staged)} file(s):")
    for name in staged[:20]:
        print(f"          {name}")
    if len(staged) > 20:
        print(f"          ... and {len(staged) - 20} more")

    if args.dry_run:
        print("[dry-run] stopping before commit; run without --dry-run to publish")
        return 0

    git("commit", "-m", f"Add stage {args.stage} artefacts")
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    # Rebase, never merge: a merge commit here carries a hash that exists on no other
    # machine until it is pushed, and the branch's own instructions forbid one.
    git("pull", "--rebase", "origin", branch)
    git("push", "origin", branch)
    print(f"[ok] pushed {len(staged)} artefact(s) to origin/{branch} "
          f"as {git('rev-parse', '--short', 'HEAD').strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
