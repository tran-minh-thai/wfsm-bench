"""Print the largest rho that run_degeneracy.py --calibrate-rho marked stable.

Largest, not smallest: a heavier discount distorts the objective further than it has to,
and stability is the constraint rather than the target.

Prints nothing when the file is missing or no value is stable, which the caller reads as
"do not run the arm that needs it". A silent empty line is the right answer there -- no
stable rho means no discount in the swept range gives this dataset an interior optimum
that survives a wider bound, and that is a result to report, not an error to work around.

Kept as a script rather than an inline one-liner because a multi-statement python -c
inside a Windows `for /f` backquote is a well-known way to lose an afternoon to quoting.

    python scripts/pick_stable_rho.py results/degeneracy_rho_qm9.csv
    python scripts/pick_stable_rho.py --report

--report reads every calibration artefact and prints one line per collection, including
the ones with no stable value. That listing is the point: the failure this guards against
is a collection quietly running under a discount nobody measured for it, and a report that
showed only the calibrated ones would hide exactly the collections at risk.
"""

import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data"


def stable_values(path: Path) -> dict[str, list[float]]:
    """dataset -> the stable rho values recorded in one artefact."""
    found: dict[str, list[float]] = {}
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError:
        return found
    for row in rows:
        if row.get("stable") == "1":
            found.setdefault(row.get("dataset", "?"), []).append(float(row["rho"]))
    return found


def report() -> int:
    calibrated: dict[str, float] = {}
    for path in sorted(RESULTS.glob("degeneracy_rho*.csv")):
        for dataset, values in stable_values(path).items():
            calibrated[dataset] = max(values + [calibrated.get(dataset, float("-inf"))])
    # Every collection with data on disk, so the ones missing a calibration are listed
    # rather than absent. A report of only what succeeded cannot show what did not.
    present = sorted(p.stem for p in DATA.glob("*.json"))
    print(f"{'collection':<24}{'calibrated rho':>16}")
    missing = 0
    for name in present:
        value = calibrated.get(name)
        if value is None:
            missing += 1
        print(f"{name:<24}{(f'{value}' if value is not None else 'none'):>16}")
    print(f"\n{len(present) - missing}/{len(present)} collections calibrated")
    if missing:
        print(f"{missing} without a stable rho: any run of theirs under discounting is "
              f"refused by the runners rather than defaulted.")
    return 0


def for_dataset(name: str) -> int:
    """Print the calibrated rho for one collection by name, or nothing.

    By name rather than by file because the caller knows which collection it is about
    and should not have to know which artefact happens to hold it. Printing nothing is
    the signal a batch `for /f` reads as "no value": the loop body never runs, so the
    caller's default stands, and a collection without a calibration drops out of the
    discounted arm instead of aborting the stage that reached it.
    """
    best = None
    for path in sorted(RESULTS.glob("degeneracy_rho*.csv")):
        for dataset, values in stable_values(path).items():
            if dataset == name:
                best = max(values + ([best] if best is not None else []))
    if best is not None:
        print(best)
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--report":
        return report()
    if argv and argv[0] == "--dataset":
        if len(argv) < 2:
            raise SystemExit("--dataset needs a collection name")
        return for_dataset(argv[1])
    path = Path(argv[0]) if argv else RESULTS / "degeneracy_rho.csv"
    values = [v for group in stable_values(path).values() for v in group]
    if values:
        print(max(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
