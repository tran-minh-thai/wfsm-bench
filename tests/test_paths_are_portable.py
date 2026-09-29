"""No code turns a path into text in a form that depends on the operating system.

The experiments run on Windows and the code is written on macOS. str() of a path gives
"results/x.csv" on one and "results\\x.csv" on the other, so any comparison, key or artefact
field built with it is right on the machine it was written on and wrong on the other. It
failed exactly that way on 2026-09-29: a check compared str(p.relative_to(...)) with
"results/x.csv", passed on macOS, and stopped the measurement machine's run.

A path becomes text through .as_posix(), or it stays a path. This scans the source for the
forms that produce OS-dependent text, and it is itself checked against each form.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each form, with why it is not portable.
FORMS = {
    "str() of a relative path": re.compile(r"\bstr\([^()]*\.relative_to\("),
    "os.sep in built text": re.compile(r"\bos\.sep\b"),
    "a path split on '/' or '\\\\'": re.compile(r"\.split\(\s*(['\"])(/|\\\\)\1\s*\)"),
}
# Display names of parametrised tests are never compared, so str() is harmless there.
ALLOWED = re.compile(r"\bids\s*=")


def offending_lines(text: str) -> list[tuple[int, str, str]]:
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        code = line.split("#", 1)[0]
        if ALLOWED.search(code):
            continue
        for name, pattern in FORMS.items():
            if pattern.search(code):
                found.append((number, name, line.strip()))
    return found


def test_no_source_file_builds_os_dependent_path_text() -> None:
    names = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    names = [n for n in names if n != "tests/test_paths_are_portable.py"]
    assert len(names) >= 50, f"only {len(names)} Python files listed; the listing is wrong"
    problems = [f"{name}:{number}: {form}: {line}"
                for name in names
                for number, form, line in offending_lines((ROOT / name).read_text(encoding="utf-8"))]
    assert not problems, ("use .as_posix() to turn a path into text:\n  "
                          + "\n  ".join(problems))


def test_the_scan_catches_each_form_and_spares_the_portable_ones() -> None:
    assert offending_lines("names = sorted(str(p.relative_to(root)) for p in files)")
    assert offending_lines("key = 'results' + os.sep + name")
    assert offending_lines("parts = path_text.split('/')")
    assert offending_lines('parts = path_text.split("\\\\")')
    assert not offending_lines("names = sorted(p.relative_to(root).as_posix() for p in files)")
    assert not offending_lines("@pytest.mark.parametrize('p', FILES, ids=lambda p: str(p.relative_to(ROOT)))")
    assert not offending_lines("# str(p.relative_to(root)) is what broke on Windows")
