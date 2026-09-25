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
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked() -> list[Path]:
    """Every file git publishes, identified by PROPERTY rather than by a list of names.

    CORRECTED 2026-09-02. The previous version scanned *.py under three named directories, which
    was 112 of the 351 tracked files -- thirty-two per cent -- while the rule it states is
    about the repository. A list of directories goes stale at exactly the moment someone adds
    a fourth, and the earlier version of this same list had already gone stale once: it named
    src/ and scripts/ only, and tests/ accumulated eighteen violating lines beside it while
    this file reported a pass.

    Asking git what is tracked cannot go stale that way. If git cannot answer, this REFUSES
    rather than falling back to the narrow list: a scan that quietly covers a third of what it
    claims is the defect this file exists to catch.
    """
    done = subprocess.run(["git", "ls-files"], cwd=ROOT,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed in {ROOT}: {done.stderr.strip()!r}. This check will not "
            f"fall back to a narrower scan, because a narrower scan is what it guards against.")
    return sorted(ROOT / name for name in done.stdout.split() if (ROOT / name).is_file())


def _readable(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        out.append(path)
    return out


# Two scopes, because the two rules have different reasons and the reason sets the scope.
#
# English and paper numbering are about PUBLISHED CONTENT, so they apply to everything the
# repository ships -- code, documents, batch files, result artefacts and their column names.
#
# The cp1252 rule is about what a redirected Windows console has to ENCODE, so it applies to
# what prints: Python and batch files. Applying it to Markdown would be a rule that does not
# match its own justification, and measured, it would fail on exactly one prose minus sign in
# docs/DesignDecisions.md that no console will ever see.
PUBLISHED_TEXT = _readable(_tracked())
PRINTS_TO_A_CONSOLE = [p for p in PUBLISHED_TEXT if p.suffix in (".py", ".bat")]
SOURCES = PUBLISHED_TEXT

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


@pytest.mark.parametrize("path",
                         [p for p in PRINTS_TO_A_CONSOLE
                          if p.name not in DEFINES_THE_PATTERNS],
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


def test_the_scan_covers_the_repository_rather_than_a_corner_of_it() -> None:
    """A parametrised check over an empty list reports a pass, having inspected nothing.

    These three rules are enforced by parametrising over a file list built at import. If that
    list came back empty -- a moved directory, a git failure, a changed layout -- pytest would
    collect zero cases per rule and the suite would be green. So the list is asserted, with
    numbers, and the numbers are printed rather than merely compared.
    """
    tracked = _tracked()
    assert len(PUBLISHED_TEXT) >= 300, (
        f"only {len(PUBLISHED_TEXT)} readable file(s) of {len(tracked)} tracked; the scan is "
        f"far narrower than the repository and these rules would pass over nothing")
    assert len(PRINTS_TO_A_CONSOLE) >= 100, (
        f"only {len(PRINTS_TO_A_CONSOLE)} file(s) that print to a console; the encoding rule "
        f"would pass over nothing")
    # Every published extension is inside the content scope: catching a new kind of file by
    # property means nobody has to remember to add it.
    uncovered_kinds = sorted({p.suffix for p in tracked if p.is_file()} - {p.suffix for p in PUBLISHED_TEXT})
    assert not uncovered_kinds, f"tracked file kinds excluded from the content rules: {uncovered_kinds}"


def test_the_rules_reject_what_they_forbid() -> None:
    """A guard that has never refused anything has never guarded anything.

    Checked in BOTH directions: a rule that matches everything refuses nothing useful, and a
    rule that matches nothing refuses nothing at all. This file is the one place allowed to
    contain the characters it forbids, so the positive cases can be written literally.
    """
    # Vietnamese: hits a diacritic, ignores plain ASCII.
    assert VIETNAMESE.search("H\u1ed1i ti\u1ebfc trung b\u00ecnh"), "a diacritic was not seen"
    assert not VIETNAMESE.search("mean regret over sixty-one cells"), "plain ASCII was flagged"

    # Paper numbering: hits a numbered result, ignores the object named in words.
    for sample in ("see Theorem 17", "Proposition 4 predicts", "\u00a7 3.1", "\u0110\u1ecbnh l\u00fd 17"):
        assert PAPER_NUMBERING.search(sample), f"paper numbering not seen in {sample!r}"
    assert not PAPER_NUMBERING.search("a priori bracket on the mean"), "a named object was flagged"

    # Encoding: the allowed set is COMPUTED, so five of the eleven characters source may
    # contain are correctly absent from it rather than assumed safe.
    assert CP1252_SAFE < ALLOWED_NON_ASCII, (
        "every allowed character is being treated as console-safe; the computation that "
        "distinguishes the two questions has stopped distinguishing them")
    assert "\u0394" not in CP1252_SAFE, "a character cp1252 cannot encode is listed as safe"
