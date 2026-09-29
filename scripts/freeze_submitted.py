"""Freeze the results of a submitted version into a directory that is never overwritten.

The files at the top of results/ and results-invariant/ are the current version: the next run
of a stage replaces them. At submission the numbers the manuscript reports have to stay
readable exactly as submitted, so this copies every tracked artefact into

    results/submitted/<label>_<UTC date>_<commit>/

and refuses when that directory exists, when the working tree carries uncommitted changes
(the commit in the name would then not describe the files), or when there is nothing to copy.
Earlier run directories (results/runs/) and earlier frozen versions are not copied again.

    python scripts/freeze_submitted.py --label journal-submission
    python scripts/freeze_submitted.py --label journal-submission --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEPT = ("results/runs/", "results/submitted/")


def git(*args: str) -> str:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"[fail] git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True,
                    help="what was submitted, e.g. journal-submission or revision-1")
    ap.add_argument("--dry-run", action="store_true", help="report and copy nothing")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.label):
        print(f"[refused] label {args.label!r}: use letters, digits, '.', '_' or '-'")
        return 2
    dirty = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    if dirty:
        print(f"[refused] {len(dirty)} uncommitted change(s); the commit in the directory name "
              f"would not describe the files. Commit or discard them first.")
        return 1
    trees = sorted(f"{p.name}/" for p in ROOT.iterdir()
                   if p.is_dir() and p.name.startswith("results"))
    files = [name for name in git("ls-files", "--", *trees).splitlines()
             if name and not name.startswith(KEPT)]
    if not files:
        print(f"[refused] no tracked artefact under {' '.join(trees)}; nothing to freeze")
        return 1
    commit = git("rev-parse", "--short", "HEAD").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = ROOT / "results" / "submitted" / f"{args.label}_{stamp}_{commit}"
    if target.exists():
        print(f"[refused] {target.relative_to(ROOT)} already exists and is never overwritten")
        return 1
    if args.dry_run:
        print(f"[dry-run] would copy {len(files)} file(s) into {target.relative_to(ROOT)}")
        return 0
    for name in files:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print(f"[done] {len(files)} file(s) frozen in {target.relative_to(ROOT)}; commit them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
