"""Resolve a commit recorded before the history of main was rewritten on 2026-09-29.

The rewrite removed internal documents from every commit, so every commit hash changed. Run
records written before then name the old hashes; COMMIT_MAP.tsv at the repository root lists
each old commit beside the new one that holds the same code. resolve() turns a recorded hash,
full or abbreviated, into the commit to check out today.

    python scripts/commit_map.py 942b308
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "COMMIT_MAP.tsv"


def load(path: Path = MAP) -> dict[str, str]:
    """{old full hash: new full hash}; empty when the map is absent."""
    if not path.exists():
        return {}
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split("\t")
        if len(cells) >= 2 and not line.startswith("#"):
            pairs[cells[0]] = cells[1]
    return pairs


def resolve(commit: str, path: Path = MAP) -> str:
    """The new commit for an old one, matched by prefix; the input itself when not listed.

    An abbreviation matching two old commits is refused rather than guessed.
    """
    commit = commit.strip()
    if len(commit) < 7:
        return commit
    matches = [new for old, new in load(path).items() if old.startswith(commit)]
    if len(matches) > 1:
        raise ValueError(f"{commit} matches {len(matches)} rewritten commits; give more digits")
    return matches[0] if matches else commit


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"{arg}\t{resolve(arg)}")
