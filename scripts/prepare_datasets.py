"""Download the study's datasets from TUDataset and convert them to project JSON.

Two converters, because TUDataset ships two shapes of molecular data and only one of them
is usable as it stands:

  io.tudataset  reads a collection that already carries edge attributes. Cuneiform is the
                only one that also has node labels, edge labels and sparse graphs -- see
                docs/DATASETS.md. BZR_MD is fetched too, as the degenerate-case control it
                turned out to be: its graphs are complete, at E/N 10.6.

  io.molecular  rebuilds a plain molecular collection into a weighted sparse graph, taking
                the real bonds from _A.txt and the weight from the distance between the
                bonded atoms in the coordinate columns. This is what replaces BZR_MD.

QM9 has 129433 graphs, far more than any experiment here needs, so its JSON is capped;
raise QM9_LIMIT to take more.

Idempotent: skips a dataset whose raw files already exist. Run from the project root:
    python scripts/prepare_datasets.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wfsmbench.io.molecular import convert_and_save as convert_molecular  # noqa: E402
from wfsmbench.io.tudataset import convert_and_save  # noqa: E402

BASE_URL = "https://www.chrsmrrs.com/graphkerneldatasets"
# Collections that already carry edge attributes.
DATASETS = ["BZR_MD", "Cuneiform"]
# Collections rebuilt from bonds plus coordinates (io.molecular.LAYOUTS).
MOLECULAR = ["BZR", "COX2", "DHFR", "QM9", "alchemy_full"]
# Both are far larger than anything a run reads, and both are sorted, so a prefix
# is a systematic sample rather than a random one -- noted where it matters.
QM9_LIMIT = 5000
ALCHEMY_LIMIT = 5000
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data"


def fetch(name: str) -> None:
    if (RAW / name).is_dir():
        print(f"[skip] {name}: already extracted")
        return
    url = f"{BASE_URL}/{name}.zip"
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(RAW)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        fetch(name)
        out = convert_and_save(RAW, name, OUT / f"{name}.json")
        print(f"[done] {name} -> {out}")
    for name in MOLECULAR:
        fetch(name)
        limit = {"QM9": QM9_LIMIT, "alchemy_full": ALCHEMY_LIMIT}.get(name)
        out = convert_molecular(RAW, name, OUT / f"{name}_bond.json", limit)
        print(f"[done] {name}_bond -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
