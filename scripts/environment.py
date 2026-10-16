"""Record what produced a result, so a disagreement between machines can be attributed.

Two quantities in these experiments behave completely differently across machines.

*Exhaustive enumeration is RNG-free.* Pool sizes, the optimum, the frontier, the local-optimum
count: pure functions of the data and the code. They must agree on any two machines running
the same commit, and a difference is a defect wherever it appears.

*The searches are not.* Every one draws from ``numpy.random.default_rng``, and numpy is
explicit that this is not reproducible across its own versions -- "``Generator`` does not
provide a version compatibility guarantee. In particular, as better algorithms evolve the bit
stream may change." PCG64 the bit generator is deterministic, but the methods layered on it --
``integers``, ``permutation``, ``random`` -- may consume its bits differently in a later
release. So two machines with different numpy can disagree on every search result while both
are perfectly correct.

Without knowing the versions, those two cases are indistinguishable, and a comparison that
cannot tell "our bug" from "different numpy" answers nothing. Hence this file, written beside
every run.

Besides versions and the commit, the record carries a SHA-256 of every prepared dataset in
data/, and states the memory and time limits explicitly -- "none" is written out, because a
missing line cannot be told apart from a limit nobody recorded. A runner adds its own facts
(stage, worker count) with --record.

    python scripts/environment.py                      # print
    python scripts/environment.py --out results/environment_xps.txt --record stage=17
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# numpy decides the RNG stream; networkx decides embedding enumeration order, which breaks
# ties between equally fit patterns. scipy only affects the statistics, never a search.
RNG_RELEVANT = ("numpy", "networkx")

# Deliberately absent from RNG_RELEVANT: cpu, cpu_count, memory_gb, node, platform.
# Measured 2026-08-14 -- one full cell reproduced byte-identically across arm64 and
# AMD64 under Python 3.14.6 and 3.12.10, so none of those change a reported number.
# They are recorded for timing tables and for saying which machine ran what, not
# because a search result depends on them.


def cpu_model() -> str:
    """The processor as its own operating system names it.

    platform.processor() is informative on Windows and empty or useless on macOS,
    so each platform is asked the way that actually answers. This matters only for
    timing numbers -- deterministic outputs do not depend on it -- but a timing
    table that cannot say which processor produced it is not reportable, and the
    node name alone does not say.
    """
    import subprocess
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def memory_gb() -> str:
    import os
    try:
        if platform.system() == "Windows":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total = status.ullTotalPhys
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return f"{total / 1024 ** 3:.1f}"
    except Exception:
        return "unknown"


def fingerprint() -> dict[str, str]:
    import importlib.metadata as metadata
    import os

    values = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "node": platform.node(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD")[1] or "unknown",
        "cpu": cpu_model(),
        "cpu_count": str(os.cpu_count()),
        "memory_gb": memory_gb(),
    }
    for package in ("numpy", "networkx", "scipy", "matplotlib", "pyyaml"):
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = "absent"
    dirty = uncommitted()
    values["tree"] = "clean" if not dirty else f"MODIFIED ({len(dirty)} files)"
    try:
        import subprocess
        values["commit"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        values["commit"] = "unknown"
    # Python has no heap flag and no run here carries a wall-clock limit; both are stated,
    # not omitted. A runner that does impose one overrides it with --record.
    values["memory_limit"] = "none (Python, no heap flag; each worker is its own process)"
    values["time_limit"] = "none"
    values.update(data_checksums())
    return values


def data_checksums() -> dict[str, str]:
    """SHA-256 of every prepared dataset file, keyed as data_lf.<file name>.

    Two runs on the same commit can still disagree if the data were rebuilt from a different
    download; the checksum is what tells those cases apart.

    The hash is taken with line endings normalised to LF, so the same dataset gives the same
    value on Windows and macOS. Until 2026-10-16 the converters wrote the JSON in text mode,
    which gives CRLF on Windows, and the records then hashed the raw bytes under the key
    data.<file name>: the same data had one hash on the measurement machine and another on
    the development machine. Those older records keep their key; the new key says the hash
    is of the normalised content, so the two are never compared as if they were one measure.
    """
    sums = {}
    for path in sorted((ROOT / "data").glob("*.json")):
        sums[f"data_lf.{path.name}"] = content_hash(path.read_bytes())
    return sums


def content_hash(raw: bytes) -> str:
    """SHA-256 of a text file's bytes with CRLF turned into LF."""
    import hashlib
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def uncommitted() -> list[str]:
    """Files under src/ or scripts/ that differ from what the commit hash claims.

    The commit recorded beside a run is what a reader will use to regenerate it. If
    the working tree carried edits that were never pushed, that hash describes code
    the run did not execute, and the artefact makes a claim it cannot support -- a
    worse failure than recording nothing, because nothing invites a question while a
    wrong hash closes it.
    """
    import subprocess
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    return [line[3:].strip() for line in completed.stdout.splitlines() if line.strip()]


def unpublished_artefacts(root: Path = ROOT) -> list[str] | None:
    """Files under the result trees that differ from the last commit; None if git cannot tell.

    Untracked files count and deliberately ignored ones do not. Between stages the publisher
    commits everything, so anything here was left by a stage that was stopped part-way or did
    not publish. Starting the next stage on top of it would commit those files under the next
    stage's name: on 2026-09-30 a stage was stopped by hand after writing its record and part
    of its output, and the next publish would have carried both.
    """
    import subprocess
    trees = sorted(f"{p.name}/" for p in root.iterdir() if p.is_dir() and p.name.startswith("results"))
    if not trees:
        return []
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-z", "--untracked-files=all",
             "--", *trees], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    paths, entries, index = [], completed.stdout.split("\0"), 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        if entry[0] in "RC":                  # a rename carries its old path as the next entry
            index += 1
        paths.append(entry[3:])
    return sorted(paths)


def leftover_advice(paths: list[str], root: Path = ROOT) -> list[str]:
    trees = " ".join(sorted(f"{p.name}/" for p in root.iterdir()
                            if p.is_dir() and p.name.startswith("results")))
    lines = [f"[refused] {len(paths)} file(s) under the result trees differ from the last commit:"]
    lines += [f"    {name}" for name in paths[:10]]
    if len(paths) > 10:
        lines.append(f"    ... and {len(paths) - 10} more")
    lines += [
        "Starting a stage now would publish them under that stage's commit. Decide what they are:",
        "  a stage that finished but did not publish -- publish it, with its own stage number:",
        f'    "{sys.executable}" scripts/publish_results.py --stage <its number>',
        "  a stage that was stopped part-way -- discard what it left, then run it again:",
        f"    git restore --worktree --staged -- {trees}",
        f"    git clean -f -- {trees}",
    ]
    return lines


def git(*arguments: str, timeout: int = 60) -> tuple[int, str]:
    import subprocess
    try:
        done = subprocess.run(["git", "-C", str(ROOT), *arguments],
                              capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout.strip()
    except Exception:
        return 1, ""


def behind_remote() -> tuple[str, str]:
    """(verdict, detail) for whether this checkout is the code that was pushed.

    The failure this catches is a run started from a checkout nobody has updated in
    days: the artefacts look ordinary and describe code the author has already moved
    past. It happened here -- a stage ran three commits behind and nothing in its
    output said so. A bare `git pull` does not prevent it, because it pulls whichever
    branch the machine happens to be standing on, and this repository has more than one.

    A fetch that cannot reach the network is reported as unverified, never as passed:
    "could not check" and "checked, fine" must not print the same way.
    """
    if git("fetch", "--quiet", timeout=120)[0] != 0:
        return "unverified", "could not reach the remote"
    code, upstream = git("rev-parse", "--abbrev-ref", "@{upstream}")
    if code != 0 or not upstream:
        return "unverified", "this branch tracks no remote branch"
    code, counts = git("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
    if code != 0 or not counts:
        return "unverified", f"could not compare against {upstream}"
    ahead, behind = (int(part) for part in counts.split())
    if behind and ahead:
        return "diverged", f"{ahead} ahead of and {behind} behind {upstream}"
    if behind:
        return "behind", f"{behind} commits behind {upstream}"
    if ahead:
        # Ahead is the dangerous one on an experiment machine, not the safe one. Those
        # commits exist on this disk and nowhere else, so the hash recorded beside the
        # artefacts names something no reader can check out -- the artefacts become
        # unreproducible while looking perfectly documented.
        return "ahead", f"{ahead} commits ahead of {upstream}, unpushed"
    return "current", f"up to date with {upstream}"


def refusal_advice(verdict: str) -> list[str]:
    """What to do about a refused checkout, one cause per message.

    The old message sent every refusal to the development machine. For "behind" the fix is
    a pull right here. For "ahead" it was worse than unhelpful: the usual cause on the
    measurement machine is a publish whose push failed, so the commits hold artefacts and
    nothing else -- and since this gate runs before every stage, it blocked the very
    publish that would have pushed them, while pointing at a machine with nothing to fix.
    """
    if verdict == "behind":
        return ["The code moved on since this checkout was updated. Here: git pull --ff-only"]
    code, upstream = git("rev-parse", "--abbrev-ref", "@{upstream}")
    if code != 0 or not upstream:
        return ["This branch tracks no remote branch, so nothing can be compared."]
    _, log = git("log", "--format=%h %s", f"{upstream}..HEAD")
    commits = [line for line in log.splitlines() if line]
    _, changed = git("diff", "--name-only", f"{upstream}...HEAD")
    # An artefact tree is any top-level directory whose name starts with "results" -- the
    # same property publish_results.py stages by, so the two cannot disagree about a new tree.
    outside = [path for path in changed.splitlines()
               if path and not ("/" in path and path.split("/", 1)[0].startswith("results"))]
    lines = [f"{len(commits)} commit(s) made here and not on {upstream}:"]
    lines += [f"    {entry}" for entry in commits[:10]]
    if len(commits) > 10:
        lines.append(f"    ... and {len(commits) - 10} more")
    if outside:
        lines.append("They touch code, which leaves from the development machine only:")
        lines += [f"    {path}" for path in outside[:10]]
        lines.append("Move these commits to the development machine, or drop them here; "
                     "then: git pull --ff-only")
        return lines
    stage = next((m.group(1) for entry in commits
                  if (m := re.search(r"Add stage (\d+) artefacts", entry))), "<N>")
    publisher = Path("scripts") / "publish_results.py"
    lines.append("They hold only artefacts: a publish whose push failed. Push them, then run "
                 "the same stages again:")
    lines.append(f'    "{sys.executable}" {publisher} --stage {stage}')
    return lines


def read(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def rng_comparable(left: dict[str, str], right: dict[str, str]) -> tuple[bool, list[str]]:
    """Whether search results from these two environments can be compared at all."""
    differences = [f"{package}: {left.get(package)} vs {right.get(package)}"
                   for package in RNG_RELEVANT
                   if left.get(package) != right.get(package)]
    return not differences, differences


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require-current", action="store_true",
                    help="fetch, then exit non-zero if this checkout is behind or has "
                         "diverged from its upstream branch; a network failure is "
                         "reported as unverified, never as passed")
    ap.add_argument("--require-clean", action="store_true",
                    help="exit non-zero if src/ or scripts/ carry uncommitted changes, or "
                         "if the result trees hold files the last commit does not; run "
                         "this before a stage, so a stale or edited checkout is refused "
                         "rather than recorded under a hash that does not describe it, "
                         "and an interrupted stage's output is not published by the next")
    ap.add_argument("--record", action="append", default=[], metavar="KEY=VALUE",
                    help="add or override one line of the record, e.g. stage=17 or workers=12")
    args = ap.parse_args(argv)
    extra = {}
    for item in args.record:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            print(f"[refused] --record expects KEY=VALUE, got {item!r}")
            return 2
        extra[key.strip()] = value.strip()
    upstream = None
    if args.require_current:
        verdict, detail = behind_remote()
        if verdict in ("behind", "diverged", "ahead"):
            print(f"[refused] this checkout is {detail}.")
            print("behind: its artefacts would describe code already moved past. "
                  "ahead: they would name a commit no reader can check out. Either way "
                  "the recorded hash stops being something anyone else can verify.")
            for line in refusal_advice(verdict):
                print(line)
            return 1
        # Recorded, not merely enforced. A gate that passes leaves no trace, so an
        # artefact written when the check could not run looks exactly like one written
        # when it ran and agreed -- and "unverified" is the case worth seeing later.
        upstream = f"{verdict} ({detail})"
        print(f"[{'ok' if verdict == 'current' else 'warn'}] checkout {verdict}: {detail}")
    if args.require_clean:
        dirty = uncommitted()
        if dirty:
            print("[refused] uncommitted changes under src/ or scripts/:")
            for name in dirty[:10]:
                print(f"    {name}")
            if len(dirty) > 10:
                print(f"    ... and {len(dirty) - 10} more")
            print("The commit hash written beside this run would not describe the code "
                  "that ran. Commit and push from the development machine, then pull "
                  "here, then run.")
            return 1
        leftover = unpublished_artefacts()
        if leftover is None:
            print("[refused] git could not report the state of the result trees, so files left "
                  "by an interrupted stage cannot be ruled out. Run `git status -- results` "
                  "here and fix what it reports.")
            return 1
        if leftover:
            for line in leftover_advice(leftover):
                print(line)
            return 1
    values = fingerprint()
    if upstream is not None:
        values["upstream"] = upstream
    values.update(extra)
    text = "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[done] {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
