"""Find checks that are narrower than the claim they stand behind.

This defect class has hit this project repeatedly, and every instance looked like a passing
test:

  a parallel-edge count run on a structure that keys edges by an unordered pair, so it could
  not have reported anything but zero;
  a search limited to one file extension standing behind a judgement about all code;
  a test whose expected value was read from the very function it was checking, so removing a
  column removed it from the expectation too;
  a comparison over an empty intersection printing "0 disagreements".

None of these can be found by reading, because the code says what its author meant. They are
found by asking, mechanically, whether the thing being measured could have come out other
than it did.

Four signals, each a heuristic. The output is a WORK LIST to verify by hand, not a verdict:
a flagged check may be fine, and an unflagged one may still be too narrow. The count of
checks it could not classify is printed, because a scan that inspects nothing and reports
nothing wrong is the same failure it is looking for.

    python scripts/audit_check_scope.py
    python scripts/audit_check_scope.py --signal vacuous
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
# The work list is a count over source text, so it does not depend on the machine and belongs
# with the other invariant artefacts. It is written rather than printed because a figure this
# tool produces has already been quoted in the project journal, and a quoted number with no
# artefact behind it cannot be checked later.
OUT = ROOT / "results-invariant" / "check_scope_audit.csv"

# Module-level scans are flagged too, since a file-wide SOURCES list bounds every check in
# the file at once -- which is how a publishability check came to cover only two directories.
SIGNALS = {
    "vacuous": "asserts over a collection that may be empty, with no check that it is not",
    "self-referential": "builds its expectation from the module it is testing",
    "hardcoded": "compares against a literal rather than recomputing from the definition",
    "narrowed-scan": "restricts a search by extension or directory while claiming more",
}

# Names that mean "the set I am asserting over"; emptiness makes the assertion vacuous.
COLLECTION_CALLS = {"len", "sum", "any", "all", "sorted", "set", "list"}
# A guard that rules the vacuous case out.
GUARD_HINTS = ("assert ", "pytest.skip", "denominator", "vacuous", "at least one",
               "would be vacuous", "nothing to")


def functions(path: Path) -> list[tuple[str, ast.FunctionDef, str]]:
    text = path.read_text(encoding="utf-8")
    tree, source = ast.parse(text), text.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            body = "\n".join(source[node.lineno - 1:node.end_lineno])
            out.append((node.name, node, body))
    return out


def called_names(node: ast.AST) -> set[str]:
    """Every function name called anywhere under this node."""
    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            out.add(getattr(func, "id", None) or getattr(func, "attr", ""))
    return out - {""}


def module_scope_bound(text: str, tree: ast.Module) -> str:
    """A scope restriction at module level, which bounds EVERY check in the file at once.

    This is the worse form and the one the first version missed: a file-wide SOURCES list
    built with rglob("*.py") means every assertion in that file is about Python files only,
    however global the claim in its docstring. Function bodies alone never see it.
    """
    lines = text.splitlines()
    inside = {n.lineno for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              for n in [node] for _ in [0]}
    spans = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for number, line in enumerate(lines, 1):
        if any(start <= number <= end for start, end in spans):
            continue
        if any(w in line for w in ('rglob("*.', 'glob("*.', "--include", 'endswith(".')):
            return line.strip()
    return ""


def tested_names(tree: ast.Module) -> set[str]:
    """Names this file pulls in from the code under test, by import or by loading a script."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("wfsmbench"):
            out.update(alias.asname or alias.name for alias in node.names)
    return out


def flags(node: ast.FunctionDef, body: str, tested: set[str]) -> list[str]:
    found = []
    asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]

    # An assertion over `not X` or `len(X) == 0` passes when X is empty. Unless something in
    # the function establishes X is non-empty, the check may have inspected nothing.
    over_collection = any(
        isinstance(a.test, ast.UnaryOp) and isinstance(a.test.op, ast.Not)
        or (isinstance(a.test, ast.Compare)
            and isinstance(a.test.left, ast.Call)
            and getattr(a.test.left.func, "id", "") in COLLECTION_CALLS)
        for a in asserts)
    guarded = sum(1 for hint in GUARD_HINTS if hint in body) >= 2 or len(asserts) > 1
    if over_collection and not guarded:
        found.append("vacuous")

    # The code under test appearing on BOTH sides of a comparison: the expectation is then
    # produced by the thing being checked, so breaking it moves both sides together. The
    # first version of this detector looked for the word "expected" and a module stem, and
    # missed the real instance in this repository entirely.
    for a in asserts:
        if not isinstance(a.test, ast.Compare):
            continue
        left = called_names(a.test.left) & tested
        right = set().union(*(called_names(c) for c in a.test.comparators)) & tested
        if left and right:
            found.append("self-referential")
            break
    else:
        # Also catch it when both sides are bound to locals first, which is the usual shape.
        assigned: dict[str, set[str]] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and len(child.targets) == 1:
                target = getattr(child.targets[0], "id", None)
                if target:
                    assigned[target] = called_names(child.value) & tested
        for a in asserts:
            if not isinstance(a.test, ast.Compare):
                continue
            names = [getattr(a.test.left, "id", None)] + [
                getattr(c, "id", None) for c in a.test.comparators]
            using = [assigned.get(n) for n in names if n and assigned.get(n)]
            if len(using) >= 2:
                found.append("self-referential")
                break

    # A literal expectation is right on a small hand-counted fixture and wrong when it
    # encodes a belief about a derived result. The artefact directory is what separates the
    # two, so only literals in a check that reads results/ are flagged.
    literal_compare = any(
        isinstance(a.test, ast.Compare) and any(isinstance(c, ast.Constant)
                                                and not isinstance(c.value, bool)
                                                for c in a.test.comparators)
        for a in asserts)
    reads_artefacts = any(w in body for w in ("RESULTS", "results/", "INVARIANT",
                                              "results-invariant", ".csv"))
    if literal_compare and reads_artefacts and "approx" not in body:
        found.append("hardcoded")

    # A search bounded by extension or directory, standing behind a claim about all code.
    if any(w in body for w in ("--include", "rglob(", "endswith(\".", "suffix ==",
                               "glob(\"*.")):
        found.append("narrowed-scan")
    return found


def provenance() -> dict[str, str]:
    """What produced this list. Without the tool's own hash, an argument about an old count
    cannot end: nobody can tell a wrong number from a number this tool no longer computes."""
    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""
    source = Path(__file__).read_bytes()
    dirty = git("status", "--porcelain")
    return {
        "tool": Path(__file__).name,
        "tool_sha256": hashlib.sha256(source).hexdigest()[:16],
        "commit": git("rev-parse", "--short", "HEAD"),
        "tree": "clean" if not dirty else f"MODIFIED ({len(dirty.splitlines())} files)",
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal", choices=sorted(SIGNALS))
    ap.add_argument("--no-write", action="store_true",
                    help="print only; use when filtering by one signal, since a filtered "
                         "list written to the artefact would understate what was found")
    args = ap.parse_args(argv)

    checked = flagged = 0
    rows: list[tuple[str, str, list[str]]] = []
    for path in sorted(TESTS.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        tested = tested_names(tree)
        bound = module_scope_bound(text, tree)
        for name, node, body in functions(path):
            checked += 1
            found = flags(node, body, tested)
            if bound and "narrowed-scan" not in found:
                found.append("narrowed-scan")
            if args.signal:
                found = [f for f in found if f == args.signal]
            if found:
                flagged += 1
                rows.append((path.name, name, found))

    for filename, name, found in rows:
        print(f"{filename}::{name}\n    {', '.join(found)}")
    signals = sum(len(found) for _f, _n, found in rows)
    print(f"\n[scope] {checked} check(s) inspected, {flagged} check(s) flagged, "
          f"{signals} flag(s) in total -- a check may carry more than one")
    if checked == 0:
        print("[refused] nothing was inspected, which is not the same as nothing being wrong",
              file=sys.stderr)
        return 2
    for signal in sorted(SIGNALS):
        count = sum(1 for _f, _n, found in rows if signal in found)
        print(f"    {signal:<17} {count:>3}  -- {SIGNALS[signal]}")
    print("\nThis is a work list, not a verdict. Verify each by asking: could this check have "
          "come out\nother than it did, if the thing it guards were broken?")

    if args.signal or args.no_write:
        print("[note] nothing written: a list filtered to one signal would understate the "
              "sweep")
        return 0

    where = provenance()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "check", "signal", "checks_inspected",
                         "checks_flagged", *where])
        for filename, name, found in rows:
            for signal in found:
                writer.writerow([filename, name, signal, checked, flagged,
                                 *where.values()])
    print(f"[ok] {OUT.relative_to(ROOT)} -- {flagged} flagged of {checked} inspected, "
          f"tool {where['tool_sha256']}, tree {where['tree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
