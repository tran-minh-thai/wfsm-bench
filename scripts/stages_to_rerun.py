r"""Print the stages whose artefacts cannot be traced to a commit in this repository.

Each stage writes results/environment_stage<N>.txt naming the commit it ran from. A commit
outside the history of HEAD -- one made before the history was compressed on 2026-08-25 --
cannot be checked out from a clone, so nobody can say which code produced that stage's
numbers. Re-running the stage from the current commit fixes that, and because the outputs
are deterministic, any number that changes on the re-run is a finding in its own right.

The set is computed, not typed. A list written out by hand goes stale the moment one stage is
re-run, and a runner fed a stale list re-measures cells that were already done.

    python scripts/stages_to_rerun.py            stage numbers on stdout, reasons on stderr

Order: the calibration stages first, because later stages read the rho they pick; the
determinism check last, so a drift it finds does not cost the stages after it.

Exit codes: 0 a set was printed; 3 nothing to re-run; 2 refused (no record was read at all,
which is not the same as every record being traceable).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNNER = ROOT / "run_experiments_windows.bat"

RECORD = re.compile(r"environment_stage(\d+)\.txt")
# Stages that calibrate rho. The runner's header says they must be re-run together, since
# running one leaves the artefacts on two different sweeps; a test holds this tuple to that
# header so the two cannot drift apart.
CALIBRATION = (1, 3, 15)
# The determinism check compares stage 13's output with the runs of stages 2, 4, 9 and 12
# (the pairs are in the runner's :stage14). It follows any of them that is re-run, and goes
# last because a drift makes it stop the suite.
CHECKS = {14: (2, 4, 9, 12, 13)}

NOTHING_TO_DO = 3
REFUSED = 2
# Stands in for the commit of a stage whose artefacts are present but which has no record.
UNRECORDED = "<no record>"


def stage_outputs() -> dict[int, list[str]]:
    """{stage: glob patterns, relative to the results directory, of the logs it writes}.

    Read from the runner through scripts/stage_commands.py, so a stage added to the runner
    is covered without a list here to keep current.
    """
    import re as _re
    spec = importlib.util.spec_from_file_location(
        "stage_commands", Path(__file__).resolve().parent / "stage_commands.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = {}
    for stage, commands in module.stage_commands().items():
        patterns = set()
        for command in commands:
            for target in _re.findall(r">>?\s*results/(\S+)", command):
                patterns.add(_re.sub(r"\$\{\w\}", "*", target))
        outputs[stage] = sorted(patterns)
    return outputs


def read_commit(path: Path) -> str | None:
    """The commit a record names. Records written on Windows keep their CRLF (results*/** is
    -text), so the line is stripped: a trailing carriage return made every commit look
    unknown to git in the first version of this check."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "commit":
            return value.strip() or None
    return None


def records(results: Path) -> dict[int, str | None]:
    """{stage: commit named in its record}. Only environment_stage<N>.txt counts; a file such
    as environment_stage16_n267_superseded.txt is a kept-for-reference copy, not a stage."""
    out = {}
    for path in results.glob("environment_stage*.txt"):
        match = RECORD.fullmatch(path.name)
        if match:
            out[int(match.group(1))] = read_commit(path)
    return out


def untraceable_reason(commit: str | None) -> str | None:
    """None when the commit is in the history of HEAD; otherwise why it is not."""
    if commit == UNRECORDED:
        return "its artefacts are present but it has no environment_stage<N>.txt manifest"
    if not commit:
        return "record names no commit"
    known = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
                           capture_output=True)
    if known.returncode != 0:
        return f"{commit} is not in this repository (made before the 2026-08-25 compression)"
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                              cwd=ROOT, capture_output=True)
    if ancestor.returncode != 0:
        return f"{commit} exists here but is not in the history of HEAD"
    return None


def select(found: dict[int, str | None],
           reason: Callable[[str | None], str | None]) -> tuple[list[int], dict[int, str]]:
    """(stages in run order, {stage: why it is in the set})."""
    why = {stage: r for stage, commit in found.items() if (r := reason(commit)) is not None}
    if any(stage in why for stage in CALIBRATION):
        for stage in CALIBRATION:
            why.setdefault(stage, "calibrates together with another stage in this set")
    for check, compared in CHECKS.items():
        rerun = [stage for stage in compared if stage in why]
        if rerun:
            why.setdefault(check, f"compares the output of stage(s) "
                                  f"{', '.join(map(str, rerun))}, which re-run")
    first = [s for s in CALIBRATION if s in why]
    last = [s for s in sorted(CHECKS) if s in why]
    middle = sorted(s for s in why if s not in first and s not in last)
    return first + middle + last, why


def main(argv: list[str] | None = None,
         reason: Callable[[str | None], str | None] = untraceable_reason) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    found = records(args.results)
    if not found:
        print(f"[refused] no environment_stage<N>.txt under {args.results}; reading nothing "
              f"is not the same as every stage being traceable", file=sys.stderr)
        return REFUSED

    # A stage whose logs are here but whose manifest is not ran under a commit nobody can
    # name. Before 2026-09-29 such a stage was only mentioned in a note, on the grounds that
    # "nothing says it was ever run" -- while its artefacts sat in results/ and were read.
    recorded = len(found)
    for stage, patterns in stage_outputs().items():
        if stage not in found and any(any(args.results.glob(p)) for p in patterns):
            found[stage] = UNRECORDED
    wrapped = lambda commit: (untraceable_reason(commit) if commit == UNRECORDED
                              else reason(commit))
    order, why = select(found, wrapped)
    print(f"[read] {recorded} stage record(s) under {args.results}", file=sys.stderr)
    for stage in order:
        print(f"  stage {stage:>2}: {why[stage]}", file=sys.stderr)
    if not order:
        print(f"[nothing] all {recorded} record(s) name a commit in the history of HEAD, "
              f"and every stage with artefacts has one",
              file=sys.stderr)
        return NOTHING_TO_DO
    print(" ".join(map(str, order)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
