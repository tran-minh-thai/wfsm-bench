"""README.md may not tell anyone to run a script that is gone, and its generated sections
must match their sources.

Measured 2026-09-04 on the old Windows guide: three of the four scripts it named had been
removed when the suite moved to a staged entry point, and nothing failed. Measured again on
2026-09-29 on README.md itself: nine of the scripts in its command table no longer existed.
A guide is read by someone who cannot tell a stale instruction from a current one.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Paths appear in POSIX or Windows form. The raw string keeps the backslash from being read
# as an escape -- reading it as one is what made a first attempt at this check report every
# name as missing.
NAMED = re.compile(
    r"(scripts[\\/][A-Za-z0-9_]+\.py|(?<![\\/])\brun_[A-Za-z0-9_]+\.(?:py|bat)|tests[\\/][A-Za-z0-9_]+\.py)")
LINKED = re.compile(r"\]\(([^)#\s]+)\)")


def test_every_script_the_readme_names_exists() -> None:
    text = README.read_text(encoding="utf-8")
    named = {m.replace("\\", "/") for m in NAMED.findall(text)}
    assert len(named) >= 10, (
        f"only {len(named)} script name(s) found in README.md; the pattern stopped matching "
        f"and this check would pass over everything")
    missing = sorted(n for n in named
                     if not (ROOT / n).exists() and not (ROOT / "scripts" / n).exists())
    assert not missing, f"README.md names {len(missing)} script(s) that do not exist: {missing}"


def test_every_relative_link_in_the_readme_resolves() -> None:
    text = README.read_text(encoding="utf-8")
    links = [link for link in LINKED.findall(text) if "://" not in link]
    assert links, "no relative link found; the pattern stopped matching"
    broken = sorted(link for link in links if not (ROOT / link).exists())
    assert not broken, f"README.md links to paths that do not exist: {broken}"


def test_the_generated_sections_match_their_sources() -> None:
    done = subprocess.run([sys.executable, "scripts/readme_sections.py", "--check"], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_currency_check_can_fail(tmp_path: Path, monkeypatch) -> None:
    """A stale section must be reported: change one generated line and run the check."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("readme_sections",
                                                  ROOT / "scripts" / "readme_sections.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = README.read_text(encoding="utf-8")
    assert "| `alpha` | `1.0` |" in text, "the parameter table is not where it was"
    stale = tmp_path / "README.md"
    stale.write_text(text.replace("| `alpha` | `1.0` |", "| `alpha` | `9.0` |", 1),
                     encoding="utf-8")
    monkeypatch.setattr(module, "README", stale)
    assert module.main(["--check"]) == 1
