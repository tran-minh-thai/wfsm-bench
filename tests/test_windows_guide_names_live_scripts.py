"""The Windows guide may not tell the experiment machine to run a script that is gone.

Measured 2026-09-04: three of the four scripts the guide named had been removed when the suite
moved to a staged entry point, and nothing failed. A guide is read on the machine furthest from
the code, by someone who cannot tell a stale instruction from a current one, so a dead name
there costs a trip to a machine rather than a diff.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RUN_ON_WINDOWS.md"

# Paths appear in Windows form inside fenced commands. The raw string keeps the backslash from
# being read as an escape -- reading it as one is what made a first attempt at this check
# report every name as missing.
NAMED = re.compile(
    r"(scripts[\\/][A-Za-z0-9_]+\.py|(?<![\\/])\brun_[A-Za-z0-9_]+\.(?:py|bat))")

# A name inside this section is a record of what was REMOVED, so it must not be resolved.
HISTORY = "**Earlier versions of this guide"


def test_every_script_the_guide_names_exists() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    cut = text.index(HISTORY) if HISTORY in text else len(text)
    live = text[:cut]

    named = {m.replace("\\", "/") for m in NAMED.findall(live)}
    assert len(named) >= 2, (
        f"only {len(named)} script name(s) found in the guide; the pattern stopped matching "
        f"and this check would pass over everything")

    missing = sorted(n for n in named
                     if not (ROOT / n).exists() and not (ROOT / "scripts" / n).exists())
    assert not missing, (
        f"{len(missing)} of {len(named)} script(s) the guide tells the experiment machine to "
        f"run do not exist: {missing}")


def test_the_history_note_is_excluded_rather_than_resolved() -> None:
    """Otherwise the note recording the removal would itself fail the check above."""
    text = GUIDE.read_text(encoding="utf-8")
    assert HISTORY in text, "the history note is gone; the exclusion above now cuts nothing"
    removed = NAMED.findall(text[text.index(HISTORY):])
    assert removed, "the history note names no script, so it records nothing"
    assert all(not (ROOT / n.replace("\\", "/")).exists()
               and not (ROOT / "scripts" / n).exists() for n in removed), (
        "a script named in the history note as removed is present in the repository; either "
        "it came back, or the note is wrong")
