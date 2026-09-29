"""Write the generated sections of README.md: datasets, parameters, and stage commands.

Each section is read from its source rather than typed: the dataset table from
results-invariant/dataset_profile.csv, the parameter table from src/wfsmbench/config.py and the
constants listed below, the stage commands from run_experiments_windows.bat. A hand-typed copy
of any of them goes stale the first time the source changes, and nothing would say so;
tests/test_readme_is_current.py fails instead.

    python scripts/readme_sections.py            # rewrite the sections in README.md
    python scripts/readme_sections.py --check    # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
sys.path.insert(0, str(ROOT / "src"))

# Module-level constants that change a reported number, with what each one controls. The
# value column is read from the source, so only the description is written here.
CONSTANTS = [
    ("src/wfsmbench/config.py", "STUDY_SIZE_REFERENCE",
     "fitness normalisation anchor (pattern edges) used by every experiment script"),
    ("src/wfsmbench/io/tudataset.py", "WEIGHT_MIN", "lower end of the per-dataset edge-weight scaling"),
    ("src/wfsmbench/io/tudataset.py", "WEIGHT_MAX", "upper end of the per-dataset edge-weight scaling"),
    ("scripts/prepare_datasets.py", "QM9_LIMIT", "graphs kept from QM9 when the dataset is built"),
    ("scripts/prepare_datasets.py", "ALCHEMY_LIMIT", "graphs kept from alchemy_full when built"),
    ("src/wfsmbench/io/protein.py", "CUTOFF", "contact distance (angstrom) for the protein contact graphs"),
    ("src/wfsmbench/pso/optimizer.py", "STALL_ROUNDS", "consecutive rounds without a new pattern before any search restarts part of its population"),
    ("src/wfsmbench/baselines/local_search.py", "PERTURBATION", "random edits applied by an iterated-local-search kick"),
    ("scripts/run_weight_mode.py", "SWARM", "population size of every population-based method"),
    ("scripts/run_weight_mode.py", "BUDGETS", "evaluation budgets (distinct patterns scored) per run"),
    ("scripts/run_weight_mode.py", "SUPPORT_FRACTIONS", "support thresholds, as a fraction of the graphs used"),
    ("scripts/run_degeneracy.py", "RHOS", "discount factors swept when rho is calibrated"),
    ("scripts/run_degeneracy.py", "GAMMAS", "size-penalty coefficients swept by the degeneracy analysis"),
    ("scripts/run_pareto_size.py", "SUPPORT_FRACTIONS", "support thresholds of the Pareto-front runs"),
    ("scripts/run_pareto_size.py", "STUDY_GRAPH_LIMIT", "graphs used from Cuneiform in the Pareto-front runs"),
    ("scripts/run_statistics.py", "ALPHA", "significance level of the pairwise tests (Holm-corrected)"),
    ("scripts/run_selector.py", "FIXED", "fixed single-method rules the racing selector is compared with"),
    ("src/wfsmbench/measure/selector.py", "DECOMPOSITION_SIZE", "race size at which the regret bound's two terms are separated"),
    ("scripts/measure_race_only_signals.py", "TIE", "fitness gap under which two methods count as tied in a race"),
    ("scripts/make_figures.py", "CD", "Nemenyi critical difference drawn on the rank figures (7 methods, 80 blocks, alpha 0.05)"),
]

# The collections each dataset row comes from, for the table's last column.
SOURCES = {
    "BZR_MD": "TUDataset, edge distances",
    "Cuneiform": "TUDataset, edge attributes",
    "pdb_contact": "protein contact graphs (tracked, not rebuilt)",
}


def _number(node: ast.AST):
    """A literal, or simple arithmetic on literals such as 1 / 30."""
    if isinstance(node, ast.BinOp):
        left, right = _number(node.left), _number(node.right)
        operations = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                      ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}
        return operations[type(node.op)](left, right)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_number(element) for element in node.elts]
    return ast.literal_eval(node)


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|")


def _constant(path: str, name: str) -> str:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == name and node.value is not None:
            try:
                value = _number(node.value)
            except (ValueError, KeyError):
                return ast.unparse(node.value)
            if isinstance(value, float):
                return f"{value:g}"
            if isinstance(value, (list, tuple)):
                return ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in value)
            return str(value)
    raise SystemExit(f"[fail] {path} defines no module-level {name}")


def _field_notes() -> dict[str, str]:
    """{Params field: first sentence of the comment block above it (or above its group)}."""
    lines = (ROOT / "src" / "wfsmbench" / "config.py").read_text(encoding="utf-8").splitlines()
    notes, block, inside = {}, [], False
    for line in lines:
        if line.startswith("class Params"):
            inside = True
            continue
        if not inside:
            continue
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("_"):
            break
        if stripped.startswith("#"):
            if not block or block[-1] is None:
                block = []
            block.append(stripped.lstrip("# ").strip())
            continue
        field = re.match(r"(\w+)\s*:\s*[\w\[\], |]+=", stripped)
        if field:
            text = " ".join(b for b in block if b)
            sentence = re.split(r"(?<=\.)\s", text, maxsplit=1)[0] if text else ""
            if len(sentence) > 170:
                sentence = sentence[:167].rsplit(" ", 1)[0] + " ..."
            notes[field.group(1)] = sentence
            block.append(None)
    return notes


def parameters() -> str:
    from wfsmbench.config import Params
    notes = _field_notes()
    rows = ["| parameter | default | what it sets |", "|---|---|---|"]
    for field in dataclasses.fields(Params):
        default = field.default
        shown = f"`{default!r}`" if isinstance(default, str) else f"`{default}`"
        rows.append(f"| `{field.name}` | {shown} | {_cell(notes.get(field.name, ''))} |")
    rows += ["", "| constant | file | value | what it sets |", "|---|---|---|---|"]
    for path, name, meaning in CONSTANTS:
        rows.append(f"| `{name}` | `{path}` | `{_constant(path, name)}` | {_cell(meaning)} |")
    return "\n".join(rows) + "\n"


def datasets() -> str:
    path = ROOT / "results-invariant" / "dataset_profile.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"[fail] {path} holds no dataset rows")
    out = ["| dataset | graphs | mean nodes | mean edges | node labels | edge labels | "
           "mean weight | source |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        name = row["dataset"]
        source = SOURCES.get(name, "TUDataset, rebuilt from bonds and atom coordinates")
        out.append(f"| {name} | {row['graphs']} | {float(row['nodes_mean']):.1f} | "
                   f"{float(row['edges_mean']):.1f} | {row['node_labels']} | "
                   f"{row['edge_labels']} | {float(row['weight_mean']):.3f} | {source} |")
    return "\n".join(out) + "\n"


def stage_commands() -> str:
    spec = importlib.util.spec_from_file_location("stage_commands",
                                                  ROOT / "scripts" / "stage_commands.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.markdown()


SECTIONS = {"datasets": datasets, "parameters": parameters, "stage-commands": stage_commands}


def render(text: str) -> str:
    for name, build in SECTIONS.items():
        begin = f"<!-- generated:{name} -->"
        end = f"<!-- /generated:{name} -->"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise SystemExit(f"[fail] README.md must hold exactly one {begin} ... {end} pair")
        head, rest = text.split(begin)
        _, tail = rest.split(end)
        text = f"{head}{begin}\n{build()}{end}{tail}"
    return text


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report, and write nothing")
    args = ap.parse_args(argv)
    current = README.read_text(encoding="utf-8")
    wanted = render(current)
    if args.check:
        if wanted != current:
            print("[stale] README.md differs from its sources; run scripts/readme_sections.py")
            return 1
        print(f"[ok] README.md matches its sources ({len(SECTIONS)} generated sections)")
        return 0
    if wanted != current:
        README.write_text(wanted, encoding="utf-8")
        print("[done] README.md regenerated")
    else:
        print("[ok] README.md already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
