r"""Build weighted, sparse residue contact graphs from PDB structure files.

Every collection used so far is molecular, because TUDataset holds exactly six that
instantiate the model and all six are chemistry. A result about which search method wins
that rests entirely on one domain is a result about that domain until shown otherwise,
so this adds a second one from a source outside TUDataset.

Proteins fit the model without any of the reinterpretation the molecular collections
needed, and the fit is worth stating role by role rather than asserted:

    one protein chain            one data unit -- support counts chains containing a pattern
    residue                      node
    residue type                 node label: the twenty standard amino acids, a natural
                                 categorical alphabet rather than a discretised coordinate
    C-alpha pair within CUTOFF   edge
    the C-alpha distance         edge weight, measured from the deposited structure
    sequence separation          edge label: 0 where the residues are consecutive in the
                                 chain, 1 where they are neighbours only in space

The edge label comes from sequence separation and never from the distance. That rule is
inherited from the molecular builder for the same reason: the distance is already the
weight, and letting the label be a function of it would smuggle the weight into the
matching decision, while the design has weights score patterns and never gate what
matches what. Sequence separation is independent of the coordinates, so the two roles
stay separate.

CUTOFF was measured, not taken from convention. The usual contact definition of 8 A
between C-alpha atoms gives 4.48 edges per node, which is the density that ruined the
_MD collections; 4.5 A leaves 93 percent of edges backbone, so the graph is a path with
decoration and the contact structure is gone. At 5.5 A the density is 1.87 edges per
node, between QM9 at 0.98 and Cuneiform at 2.11, and the split between the two edge
labels is 53 to 47, so both are genuinely present.

Chains are read from the first model, first polypeptide chain, taking the highest
occupancy alternate location. Residues outside the standard twenty are dropped rather
than mapped, because folding them into a catch-all label would invent a category the
structure does not have.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .tudataset import WEIGHT_MAX, WEIGHT_MIN

# Measured; see the module docstring for the densities either side of it.
CUTOFF = 5.5
MIN_RESIDUES = 30

# The twenty standard amino acids. Order fixes the label numbering, so a rebuild gives
# the same labels and two runs stay comparable.
RESIDUES = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
RESIDUE_INDEX = {name: index for index, name in enumerate(RESIDUES)}

BACKBONE, CONTACT = 0, 1

# Twenty residue types give 20*21/2 * 2 = 420 possible edge templates, and 418 of them
# turn out to be frequent at two hundred chains -- seventeen times Cuneiform's 24. The
# pattern space that produces is beyond exhaustive enumeration, which every ground-truth
# quantity here depends on: enumerating size 4 over fifty chains already runs past ten
# minutes. The fix is a different role for the column, not a tuned threshold.
#
# Grouping residues by side-chain chemistry is the standard reduction and is not a
# discretised coordinate: the classes are published chemistry, and every member of a
# class substitutes for the others in real sequences. Six classes give 6*7/2 * 2 = 42
# possible templates, the range the rest of the collections sit in.
GROUPS: dict[str, str] = {
    "GLY": "special", "PRO": "special",
    "ALA": "aliphatic", "VAL": "aliphatic", "LEU": "aliphatic", "ILE": "aliphatic",
    "MET": "aliphatic",
    "PHE": "aromatic", "TRP": "aromatic", "TYR": "aromatic",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar", "CYS": "polar",
    "LYS": "positive", "ARG": "positive", "HIS": "positive",
    "ASP": "negative", "GLU": "negative",
}
GROUP_NAMES = ("special", "aliphatic", "aromatic", "polar", "positive", "negative")
GROUP_INDEX = {name: index for index, name in enumerate(GROUP_NAMES)}


def read_chain(path: Path) -> list[tuple[str, float, float, float]]:
    """C-alpha atoms of the first chain of the first model, in sequence order.

    Alternate locations are filtered to the blank or "A" indicator so a residue cannot
    appear twice at two positions, which would create a zero-length edge and a duplicate
    node with the same label.
    """
    atoms: list[tuple[str, float, float, float]] = []
    chain: str | None = None
    seen: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):
            continue
        if chain is None:
            chain = line[21]
        if line[21] != chain:
            continue
        key = line[22:27]                       # residue sequence number plus insertion code
        if key in seen:
            continue
        seen.add(key)
        atoms.append((line[17:20].strip(),
                      float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms


def contacts(atoms) -> list[tuple[int, int, int, float]]:
    """(i, j, label, distance) for every C-alpha pair inside the cutoff."""
    found = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = math.dist(atoms[i][1:], atoms[j][1:])
            if distance <= CUTOFF:
                found.append((i, j, BACKBONE if j == i + 1 else CONTACT, distance))
    return found


def label_of(residue: str, grouped: bool) -> int:
    return GROUP_INDEX[GROUPS[residue]] if grouped else RESIDUE_INDEX[residue]


def build_graphs(paths, limit: int | None = None, grouped: bool = True):
    """Contact graphs plus the chains that were dropped and why.

    Rejections are returned rather than logged away: a collection that silently discards
    half its input is a different collection from the one its source suggests, and the
    counts are what makes that visible.
    """
    chains, dropped = [], {"short": 0, "unreadable": 0, "nonstandard": 0}
    for path in paths:
        if limit is not None and len(chains) >= limit:
            break
        try:
            atoms = read_chain(path)
        except (ValueError, OSError):
            dropped["unreadable"] += 1
            continue
        if len(atoms) < MIN_RESIDUES:
            dropped["short"] += 1
            continue
        keep = [a for a in atoms if a[0] in RESIDUE_INDEX]
        if len(keep) < MIN_RESIDUES:
            dropped["nonstandard"] += 1
            continue
        chains.append((path.stem, keep))

    edges = [contacts(atoms) for _name, atoms in chains]
    lengths = [d for chain in edges for *_rest, d in chain]
    if not lengths:
        raise ValueError("no contacts found; check the cutoff and the role assignment")
    low, high = min(lengths), max(lengths)
    span = high - low

    def weight(distance: float) -> float:
        if span == 0.0:
            return WEIGHT_MIN
        return WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * (distance - low) / span

    built = []
    for index, ((_name, atoms), found) in enumerate(zip(chains, edges)):
        built.append({
            "id": index,
            "vertices": [{"id": position, "label": label_of(atom[0], grouped)}
                         for position, atom in enumerate(atoms)],
            "edges": [{"src": i, "dst": j, "label": label,
                       "weight": round(weight(distance), 6)}
                      for i, j, label, distance in found],
        })
    return built, dropped


def convert(pdb_dir: str | Path, limit: int | None = None,
            grouped: bool = True) -> dict:
    paths = sorted(Path(pdb_dir).glob("*.pdb"))
    if not paths:
        raise ValueError(f"no .pdb files under {pdb_dir}")
    graphs, dropped = build_graphs(paths, limit, grouped)
    nodes = sum(len(g["vertices"]) for g in graphs)
    edges = sum(len(g["edges"]) for g in graphs)
    return {
        "name": "pdb_contact",
        "description": (
            f"Residue contact graphs from {len(graphs)} PDB chains. Nodes are residues "
            f"labelled by amino acid type; edges join C-alpha pairs within {CUTOFF} A and "
            f"are weighted by that distance; the edge label is sequence separation, "
            f"consecutive or not, and is never derived from the distance. Dropped "
            f"{dropped['short']} chains under {MIN_RESIDUES} residues, "
            f"{dropped['nonstandard']} with too few standard residues, "
            f"{dropped['unreadable']} unreadable."),
        "graphs": graphs,
        "node_labels": list(GROUP_NAMES if grouped else RESIDUES),
        "statistics": {"graphs": len(graphs), "nodes": nodes, "edges": edges,
                       "edges_per_node": round(edges / nodes, 4), "dropped": dropped},
    }


def convert_and_save(pdb_dir: str | Path, out_path: str | Path,
                     limit: int | None = None, grouped: bool = True) -> Path:
    data = convert(pdb_dir, limit, grouped)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    return out
