"""Print the commands of each experiment stage for a POSIX shell (macOS, Linux).

run_experiments_windows.bat is the one place the stages are defined. This reads it and prints
the same commands in shell syntax, so macOS and Linux users run exactly what the Windows driver
runs, and README.md carries a section generated from this output rather than a hand copy.

    python scripts/stage_commands.py            # every stage
    python scripts/stage_commands.py 17         # one stage

Only commands are printed. Progress messages, log echoes and the driver's own bookkeeping are
left out; the driver's gates (clean checkout, manifest, publishing) are described in README.md.
Run the commands from the repository root with the virtual environment active, and set
WORKERS to the number of cores first.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "run_experiments_windows.bat"


def _joined_lines() -> list[str]:
    text = DRIVER.read_text(encoding="utf-8").replace("\r\n", "\n")
    return re.sub(r"\^\n\s*", " ", text).split("\n")


def _blocks(lines: list[str]) -> dict[str, list[str]]:
    blocks, current = {}, None
    for line in lines:
        label = re.match(r"^:(\w+)", line)
        if label:
            current = label.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line.strip())
    return blocks


def _shell(command: str, arguments: tuple[str, ...] = ()) -> str:
    """One driver command line in shell syntax."""
    for number, value in enumerate(arguments, start=1):
        command = command.replace(f"%{number}", value)
    command = re.sub(r"\s*\|\|\s*(goto :failed|exit /b 1)\s*$", "", command)
    command = command.replace("%PY%", "python").replace("\\", "/")
    command = command.replace("%WORKERS%", "$WORKERS").replace("!MODES!", '"$MODES"')
    command = re.sub(r"%%(\w)", r"${\1}", command)
    return " ".join(command.split())


def _subroutine(blocks: dict[str, list[str]], name: str, arguments: tuple[str, ...]) -> list[str]:
    out = []
    for line in blocks[name]:
        if line.startswith("%PY%"):
            out.append(_shell(line, arguments))
        elif line.startswith("echo ===") and ">>" in line:
            title, target = line[5:].split(">>", 1)
            for number, value in enumerate(arguments, start=1):
                title = title.replace(f"%{number}", value)
            out.append(f'echo "{title.strip()}" >> {target.strip().replace(chr(92), "/")}')
    return out


def stage_commands() -> dict[int, list[str]]:
    blocks = _blocks(_joined_lines())
    stages = {}
    for label, body in blocks.items():
        match = re.fullmatch(r"stage(\d+)", label)
        if not match:
            continue
        out, depth, skipping = [], 0, 0
        for line in body:
            if not line or line.upper().startswith("REM"):
                continue
            if skipping:
                skipping += line.endswith("(") - (line == ")")
                continue
            if line.startswith("if ") and line.endswith("("):
                skipping = 1
                continue
            loop = re.match(r"for %%(\w) in \(([^)]*)\) do \($", line)
            indent = "  " * depth
            if loop:
                out.append(f"{indent}for {loop.group(1)} in {' '.join(loop.group(2).split())}; do")
                depth += 1
            elif line == ")":
                depth -= 1
                out.append(f"{'  ' * depth}done")
            elif line.startswith("%PY%"):
                out.append(indent + _shell(line))
            elif line.startswith("call :modes_for "):
                dataset = _shell(line.split(maxsplit=2)[2])
                out.append(f'{indent}MODES=sum; [ -n "$(python scripts/pick_stable_rho.py '
                           f'--dataset {dataset})" ] && MODES=sum,discounted')
            elif line.startswith("call :"):
                name, *arguments = line[6:].split("||")[0].split()
                out.extend(indent + c for c in _subroutine(blocks, name, tuple(arguments)))
            elif line.startswith("del /q "):
                out.append(indent + "rm -f " + line.split()[2].replace("\\", "/"))
        if depth != 0:
            raise SystemExit(f"[fail] stage {match.group(1)}: unbalanced loop in {DRIVER.name}")
        stages[int(match.group(1))] = out
    if not stages:
        raise SystemExit(f"[fail] no :stageN block found in {DRIVER.name}")
    return dict(sorted(stages.items()))


def markdown() -> str:
    parts = []
    for number, commands in stage_commands().items():
        parts += [f"**Stage {number}**", "", "```bash", *commands, "```", ""]
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", type=int)
    args = ap.parse_args(argv)
    stages = stage_commands()
    if args.stage is not None and args.stage not in stages:
        print(f"[refused] no stage {args.stage}; stages are {', '.join(map(str, stages))}")
        return 1
    for number, commands in stages.items():
        if args.stage in (None, number):
            print(f"# stage {number}")
            print("\n".join(commands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
