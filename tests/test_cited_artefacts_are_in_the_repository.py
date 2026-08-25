"""Every artefact a generated table reads must be tracked by git.

A number in the manuscript is only checkable if whoever clones the repository can recompute
it. An artefact that exists on the author's disk and nowhere else produces a table that looks
identical and cannot be reproduced by anyone at all -- and the failure is invisible from the
author's side, because there the file is right where it should be.

It happened: a table was generated from thirteen per-seed run files, seven of which an ignore
rule kept out of the repository. Restricting to the tracked half gave eleven cells instead of
sixty-one and a different worst case. The ignore rule justified itself on size, and the size
claim was wrong by a factor of five hundred.

So this asks git what is tracked, for every results file the generators actually open.

SCOPE, stated so nobody reads more into a pass than it carries. Only text artefacts are
checked. The figures are NOT tracked, and that is deliberate: they are derived, a tracked
script reproduces them from tracked inputs, and tracking rendered binaries makes every
rebuild a diff. What guards them is test_figures_are_declared_and_drawn, which requires
every figure the draft cites to be one the builder produces.
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Generators whose output goes into a manuscript, directly or through an invariant artefact.
GENERATORS = ["sync_manuscript_numbers.py", "make_tables.py", "make_figures.py",
              "measure_race_only_signals.py", "measure_datasets.py",
              "measure_bound_locality.py", "measure_embedding_load.py"]

# Filenames and glob patterns a generator opens, however it spells them.
REFERENCE = re.compile(r'["\']([\w*./-]+\.(?:csv|log|txt))["\']')


def _tracked() -> set[str]:
    done = subprocess.run(["git", "ls-files", "results", "results-invariant"],
                          cwd=ROOT, capture_output=True, text=True)
    return set(done.stdout.split())


def _recorded_sources() -> dict[str, set[str]]:
    """Inputs an invariant artefact says it was built from.

    This is the question that matters and the only one a machine can answer honestly. A glob
    in a generator resolves against whatever the machine running it happens to hold, so
    expanding one here asks "what would this machine read", not "what produced the artefact
    that is committed". The first version expanded globs on disk and reported eleven files
    from the measurement machine's own runs as missing from the repository.
    """
    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / "results-invariant").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sources = {s for row in rows for s in (row.get("sources") or "").split() if s}
        if sources:
            found[path.name] = sources
    return found


def _named_directly() -> dict[str, set[str]]:
    """Artefacts a generator names outright, with no wildcard. Those are machine-independent
    by construction: the same literal path on every clone."""
    found: dict[str, set[str]] = {}
    for name in GENERATORS:
        path = SCRIPTS / name
        if not path.exists():
            continue
        wanted: set[str] = set()
        for token in REFERENCE.findall(path.read_text(encoding="utf-8")):
            if "*" in token:
                continue
            for directory in ("results", "results-invariant"):
                candidate = ROOT / directory / token
                if candidate.is_file():
                    wanted.add(candidate.relative_to(ROOT).as_posix())
        if wanted:
            found[name] = wanted
    return found


def test_the_scan_finds_generators_and_files() -> None:
    """Otherwise the checks below pass over nothing and still report green."""
    named, recorded = _named_directly(), _recorded_sources()
    assert named, "no generator names a results file outright; the scan is broken"
    assert recorded, ("no invariant artefact records the inputs it was built from; without "
                      "that, reproducibility cannot be checked at all")
    total = len({f for files in named.values() for f in files})
    total += len({f for files in recorded.values() for f in files})
    print(f"\n[cited] {total} artefact(s): {len(named)} generator(s) naming files outright, "
          f"{len(recorded)} artefact(s) recording their own sources")


def test_no_generator_reads_an_untracked_artefact() -> None:
    tracked = _tracked()
    assert tracked, "git reports nothing tracked under results/; cannot check"
    offenders = sorted(
        (owner, path)
        for owner, files in {**_named_directly(), **_recorded_sources()}.items()
        for path in files if path not in tracked)
    assert not offenders, (
        f"{len(offenders)} artefact(s) feed a generated table but are not in the repository, "
        f"so nobody who clones it can reproduce those numbers:\n"
        + "\n".join(f"  {g} reads {p}" for g, p in offenders[:12]))


@pytest.mark.parametrize("directory", ["results", "results-invariant"])
def test_every_invariant_artefact_is_tracked(directory: str) -> None:
    """These exist to be quoted, so an untracked one is a number with no public source."""
    base = ROOT / directory
    if directory == "results":
        pytest.skip("results/ holds working files too; the check above covers what is cited")
    tracked = _tracked()
    missing = sorted(p.relative_to(ROOT).as_posix() for p in base.glob("*")
                     if p.is_file() and p.relative_to(ROOT).as_posix() not in tracked)
    assert not missing, f"untracked artefact(s) in {directory}/: {missing}"
