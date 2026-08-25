"""Source files must be publishable as they stand: English, and free of paper numbering.

Two rules that are cheap to keep and expensive to repair in bulk, so they are checked
rather than remembered.

*English.* The repository is published; the working notes are not. Translating at
publication time is a mass edit at the busiest possible moment, with the highest chance
of changing a meaning by accident. It has already cost 138 lines across 16 files once.

This is not only a tidiness rule, which is why it is a test and not a style note. Stage
13 crashed after seven hours of searching because run_selector.py printed its summary in
Vietnamese: on Windows a redirected stdout encodes as cp1252, every diacritic raises
UnicodeEncodeError, and the run died at the last step with all its search work done and
nothing written. An ASCII check would have caught that before the machine started.

*No paper numbering.* A theorem or section number printed by a script is manuscript
content leaking into published code, and it dies the moment the paper is renumbered.
Name the object instead: "a priori bracket" rather than "Theorem 17".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Every directory that goes to the public repository, tests included. The first version
# listed src/ and scripts/ only, and the tests/ directory beside it accumulated eighteen
# violating lines -- Vietnamese constants and paper theorem numbers -- while this file
# reported a pass. A rule enforced over less than it claims is the defect it exists to catch.
PUBLISHED = ("src", "scripts", "tests")
SOURCES = sorted(path for directory in PUBLISHED
                 for path in (ROOT / directory).rglob("*.py"))

# This file must name the characters it forbids, so it cannot obey its own rule. The
# exemption is one file wide and is stated rather than implied; anything broader would let
# the next violation in beside it.
DEFINES_THE_PATTERNS = {Path(__file__).name}

# Latin letters carrying the tone and vowel marks Vietnamese uses. Catching the marks
# rather than a word list means a new phrase cannot slip through unlisted.
VIETNAMESE = re.compile(
    "[à-ãè-êìíò-õùúý"
    "ăđĩũươ"
    "Ạ-ỹ]", re.IGNORECASE)

PAPER_NUMBERING = re.compile(r"(Theorem|Lemma|Proposition|Corollary|Definition)\s+\d"
                             r"|§\s*\d"
                             r"|Định lý\s*\d")

# Non-ASCII characters allowed to appear in source at all, because they carry meaning in a
# table or a formula.
ALLOWED_NON_ASCII = set("\u00b1\u00d7\u00b7\u2013\u2014\u2265\u2264\u00c5"
                        "\u0394\u03a6\u03c1")

# Of those, the ones a redirected Windows run can actually encode -- computed, not listed.
# The list used to be one set with a comment claiming all of them survived cp1252; five of
# eleven do not, so the exemption permitted exactly the failure the rule exists to prevent.
def _cp1252_safe(characters: set[str]) -> set[str]:
    safe = set()
    for character in characters:
        try:
            character.encode("cp1252")
        except UnicodeEncodeError:
            continue
        safe.add(character)
    return safe


CP1252_SAFE = _cp1252_safe(ALLOWED_NON_ASCII)


@pytest.mark.parametrize("path", [p for p in SOURCES if p.name not in DEFINES_THE_PATTERNS],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_no_vietnamese(path: Path) -> None:
    hits = [(number, line.strip()[:70])
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if VIETNAMESE.search(line)]
    assert not hits, (
        f"{path.relative_to(ROOT)} carries Vietnamese; source is published and must be "
        f"English: {hits[:3]}")


@pytest.mark.parametrize("path", [p for p in SOURCES if p.name not in DEFINES_THE_PATTERNS],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_no_paper_numbering(path: Path) -> None:
    hits = [(number, line.strip()[:70])
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if PAPER_NUMBERING.search(line)]
    assert not hits, (
        f"{path.relative_to(ROOT)} names a numbered result from the manuscript; call the "
        f"object by name so a renumbering cannot break it: {hits[:3]}")


@pytest.mark.parametrize("path", [p for p in SOURCES if p.name not in DEFINES_THE_PATTERNS],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_output_survives_a_windows_console(path: Path) -> None:
    """Anything printable must encode as cp1252, which is what a redirect gets on Windows.

    Checked over whole lines rather than only string literals: a character in a comment
    is harmless, but distinguishing the two reliably needs a parser, and the cheap
    over-strict version costs nothing because the allowed set already covers the symbols
    a table legitimately wants.
    """
    # Against the computed cp1252 set, not the wider list of characters source may contain:
    # the two are different questions, and conflating them let five characters through that
    # cannot be encoded at all.
    offenders = sorted({c for c in path.read_text(encoding="utf-8")
                        if ord(c) > 127 and c not in CP1252_SAFE})
    assert not offenders, (
        f"{path.relative_to(ROOT)} contains {offenders}, which a redirected Windows run "
        f"cannot encode; this killed a seven-hour stage at its last step")
