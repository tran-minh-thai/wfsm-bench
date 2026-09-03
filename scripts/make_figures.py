"""Render the paper's figures from the artefacts into results/figures/.

Three figures, each carrying a claim that a table states less directly.

The previous version drew a different, since-refuted set of experiments, and it failed
the same way the table generator did: "a missing input is not a failure", so it skipped
nine of ten figures, exited zero, and left four pictures on disk whose source CSVs had
been deleted. Stale images signed with a fresh date are worse than none, so this deletes
what it is about to draw before drawing it, and refuses outright when an input is absent.

    python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.measure.ranks import mean_ranks as parse_ranks  # noqa: E402
FIGS = RESULTS / "figures"

# Axis labels, titles and legends are Vietnamese, because the draft is Vietnamese down to the
# figure labels; this file is source, which is published and stays ASCII. So the words live
# beside the manuscript and are read from there.
LABELS_PATH = ROOT.parent / "paper" / "figure_labels.vi.json"


def labels(figure: str) -> dict:
    if not LABELS_PATH.exists():
        raise SystemExit(f"[fail] missing {LABELS_PATH}. Without it every axis would be "
                         f"unlabelled or in the wrong language, and a figure nobody can read "
                         f"is worse than one that was never drawn.")
    loaded = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if figure not in loaded:
        raise SystemExit(f"[fail] {LABELS_PATH} has no labels for {figure!r}. A missing key "
                         f"must not fall back to a default: that is how a figure ships with a "
                         f"placeholder axis.")
    return loaded[figure]

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

# Nemenyi critical difference for seven methods over eighty blocks. Drawn on the sweep
# so a reader can see at a glance that nothing moves by a distinguishable amount.
CD = 1.007
METHODS = ["aco", "pso", "multiswarm", "genetic", "clpso", "local_search", "random"]
# Colourblind-safe (Okabe-Ito), fixed order so a rebuild never recolours a line.
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000", "#999999"]
QUANTA = ["q000", "q002", "q005", "q010", "q020", "q050", "q100"]
STATS_LOG = {"Cuneiform": "s5", "QM9_bond": "qm9", "BZR_bond": "BZRbond",
             "COX2_bond": "COX2bond", "DHFR_bond": "DHFRbond",
             "alchemy_full_bond": "alch"}


def need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"[fail] missing input {path.relative_to(ROOT)}. Refusing to "
                         f"draw a partial set: a figure whose data is gone is worse "
                         f"than a figure that was never drawn.")
    return path


def rows(name: str) -> list[dict]:
    with need(RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_ranks(log: str, mode: str) -> dict[str, float] | None:
    return parse_ranks(need(RESULTS / log).read_text(encoding="utf-8"), mode)

def style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_quantum(path: Path) -> None:
    """Plateau structure varies twentyfold; no method's rank follows it."""
    text = labels("dia-hinh-bi-ep-doi")
    levels, ranks = [], []
    for tag in QUANTA:
        land = rows(f"weight_mode_landscape_{tag}.csv")
        levels.append(float(land[0]["fitness_levels"]))
        got = mean_ranks(f"{tag}_statistics.log", "sum")
        ranks.append(got)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, colour in zip(METHODS, COLOURS):
        ax.plot(levels, [r[method] for r in ranks], marker="o", markersize=4,
                linewidth=1.6, color=colour, label=method)
    # The critical difference drawn against the flattest interesting line, so the
    # comparison a reader needs -- movement against the threshold of significance -- is
    # on the page rather than left to arithmetic.
    base = ranks[0]["local_search"]
    ax.axhspan(base - CD / 2, base + CD / 2, color="#999999", alpha=0.13, zorder=0)
    ax.annotate(text["cd"], xy=(levels[-1], base + CD / 2), fontsize=8,
                color="#555555", va="bottom", ha="right")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    ax.set_title(text["title"], loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_basins(path: Path) -> None:
    """Where the optimum dominates the ascent graph, ACO does not win."""
    points = []
    for source in sorted(RESULTS.glob("basins_*.csv")):
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["policy"] != "strict":
                    continue
                if row.get("graph_offset", "0") not in ("0", ""):
                    continue
                got = mean_ranks(f"{STATS_LOG[row['dataset']]}_statistics.log",
                                 row["weight_mode"])
                if got is None:
                    continue
                points.append((float(row["basin_optimum_pct"]), got["aco"],
                               row["optimum_is_largest"] == "1",
                               row["dataset"], row["weight_mode"]))
    if not points:
        raise SystemExit("[fail] no basin rows could be paired with a ranking.")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    text = labels("luu-vuc-va-dan-kien")
    for dominates, colour, label in ((True, "#D55E00", text["legend_dominates"]),
                                     (False, "#0072B2", text["legend_not"])):
        group = [p for p in points if p[2] is dominates]
        ax.scatter([p[0] for p in group], [p[1] for p in group], s=70,
                   color=colour, label=label, zorder=3,
                   edgecolor="white", linewidth=0.8)
    # Alternate the label side. Several cells sit within a few tenths of a percent of
    # each other, and a fixed offset stacked their names on top of one another.
    for index, (basin, rank, _flag, dataset, mode) in enumerate(sorted(points)):
        ax.annotate(f"{dataset.replace('_bond', '').replace('_full', '')}·{mode[:4]}",
                    (basin, rank), fontsize=7, color="#444444",
                    xytext=(5, 5) if index % 2 == 0 else (5, -10),
                    textcoords="offset points")
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.annotate("ACO first", xy=(ax.get_xlim()[1], 1.0), fontsize=8, color="#555555",
                va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    # The title states the half that holds without exception. The converse -- a
    # non-dominant optimum meaning ACO wins -- holds in six of eight, so claiming the
    # figure shows basin structure "tracking" the ranking would overstate it.
    ax.set_title(text["title"],
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


SELECTOR_ORDER = ["weighted", "ideal", "maxmin"]
SELECTOR_COLOUR = {"weighted": "#D55E00", "ideal": "#0072B2", "maxmin": "#009E73"}


def front_positions() -> dict[str, dict[str, dict[int, list[float]]]]:
    """Where each selector's pick sits on the front: 0 is the floor, 1 the maximum.

    Suffixed files are partial re-measurements of cells the unsuffixed ones already hold, so
    including them would count those cells twice.
    """
    out: dict[str, dict[str, dict[int, list[float]]]] = {}
    for source in sorted(RESULTS.glob("pareto_size_*.csv")):
        if re.search(r"_s\d", source.stem):
            continue
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                lo, hi = float(row["front_min_edges"]), float(row["front_max_edges"])
                if hi <= lo:
                    # A front spanning one edge count has no interior, so position is not
                    # defined. None was measured in any collection; if one ever appears it
                    # must be dropped visibly rather than scored as if it reached the top.
                    continue
                position = (float(row["picked_edges"]) - lo) / (hi - lo)
                bucket = out.setdefault(row["dataset"], {}).setdefault(row["selector"], {})
                bucket.setdefault(int(row["size_bound"]), []).append(position)
    if not out:
        raise SystemExit("[fail] no pareto_size_*.csv rows; nothing to draw.")
    return out


def figure_front_position(path: Path) -> None:
    """The scalarised answer is a boundary artefact; the compromise points are not."""
    text = labels("vi-tri-tren-mat")
    data = front_positions()
    names = sorted(data)
    columns = 3
    rows_of = -(-len(names) // columns)
    fig, axes = plt.subplots(rows_of, columns, figsize=(10.4, 3.2 * rows_of),
                            sharex=True, sharey=True)
    flat = axes.ravel()

    for index, dataset in enumerate(names):
        ax = flat[index]
        bounds = sorted({b for sel in data[dataset].values() for b in sel})
        for selector in SELECTOR_ORDER:
            per_bound = data[dataset].get(selector, {})
            if not per_bound:
                continue
            colour = SELECTOR_COLOUR[selector]
            for bound in bounds:
                values = per_bound.get(bound, [])
                ax.scatter([bound] * len(values), values, s=14, color=colour,
                           alpha=0.30, zorder=2, linewidth=0)
            xs = [b for b in bounds if per_bound.get(b)]
            ax.plot(xs, [statistics.mean(per_bound[b]) for b in xs], marker="o",
                    markersize=5, linewidth=1.8, color=colour, zorder=3,
                    label=text["selectors"][selector] if index == 0 else None)
        ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_title(dataset, loc="left", fontsize=9)
        ax.set_xticks(bounds)
        ax.set_ylim(-0.08, 1.12)
        style(ax)
    for spare in flat[len(names):]:
        spare.set_visible(False)

    flat[0].annotate(text["top_line"], xy=(bounds[0], 1.0), fontsize=7.5, color="#555555",
                     va="bottom", ha="left")
    flat[0].annotate(text["bottom_line"], xy=(bounds[0], 0.0), fontsize=7.5, color="#555555",
                     va="top", ha="left")
    fig.supxlabel(text["xlabel"], fontsize=9)
    fig.supylabel(text["ylabel"], fontsize=9)
    fig.suptitle(text["title"], x=0.02, ha="left", fontsize=11)
    fig.legend(frameon=False, fontsize=9, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def race_sizes() -> tuple[list[int], list[dict], list[dict]]:
    with need(RESULTS / "selector_seed_sweep.csv").open(encoding="utf-8",
                                                        newline="") as handle:
        data = list(csv.DictReader(handle))
    sizes = sorted(int(c.split("_n")[1]) for c in data[0] if c.startswith("regret_n"))
    balanced = [r for r in data if all(r[f"regret_n{n}"] for n in sizes)]
    return sizes, data, balanced


def figure_race_size(path: Path) -> None:
    """One term vanishes with more race seeds and one does not -- on a fixed set of cells.

    The two curves have DIFFERENT denominators and must not be joined. Fifty of sixty-one
    cells have no seeds past size ten, so a single mean across all sizes drops from 0.52 to
    0.16 between ten and twelve purely because the cells changed, and the eleven that remain
    are the easy ones. Drawing that as one line would show a collapse that never happened.
    """
    text = labels("hoi-tiec-theo-co-dua")
    sizes, data, balanced = race_sizes()
    complete = [n for n in sizes if len(present(data, n)) == len(data)]
    dropped = len(data) - len(balanced)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # Left: the mean total regret against the mean floor. Averaging is linear, so the gap
    # between the two lines IS the mean of the per-cell gaps -- no cross-cell subtraction.
    left = axes[0]
    full_y = [statistics.mean(present(data, n)) for n in complete]
    bal_y = [statistics.mean([float(r[f"regret_n{n}"]) for r in balanced]) for n in sizes]
    full_floor = statistics.mean([float(r["transfer_only"]) for r in data if r["transfer_only"]])
    bal_floor = statistics.mean([float(r["transfer_only"]) for r in balanced])
    left.fill_between(complete, full_y, full_floor, color="#D55E00", alpha=0.13, zorder=1,
                      label=text["gap"])
    left.plot(complete, full_y, marker="o", markersize=5, linewidth=2.0, color="#D55E00",
              zorder=4, label=text["full"].format(n=len(data)))
    left.axhline(full_floor, color="#D55E00", linewidth=1.2, linestyle="--", zorder=3,
                 label=text["floor_full"].format(n=len(data)))
    left.plot(sizes, bal_y, marker="s", markersize=4, linewidth=1.6, color="#0072B2",
              zorder=4, label=text["balanced"].format(n=len(balanced)))
    left.axhline(bal_floor, color="#0072B2", linewidth=1.0, linestyle=":", zorder=3,
                 label=text["floor_balanced"].format(n=len(balanced)))
    left.set_ylabel(text["ylabel_mean"], fontsize=9)
    left.set_title(text["panel_mean"], loc="left", fontsize=9)

    # Right: the statistical component alone -- the difference taken INSIDE each cell, then
    # maximised. Maximising regret and floor separately would subtract two different cells.
    right = axes[1]
    right.plot(complete, [max(gaps(data, n)) for n in complete], marker="o", markersize=5,
               linewidth=2.0, color="#D55E00", zorder=4,
               label=text["full"].format(n=len(data)))
    right.plot(sizes, [max(gaps(balanced, n)) for n in sizes], marker="s", markersize=4,
               linewidth=1.6, color="#0072B2", zorder=4,
               label=text["balanced"].format(n=len(balanced)))
    right.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
    right.set_ylabel(text["ylabel_worst"], fontsize=9)
    right.set_title(text["panel_worst"], loc="left", fontsize=9)

    for ax in axes:
        # The truncation is the point, so it is drawn rather than left to the caption.
        ax.axvline(complete[-1], color="#999999", linewidth=0.9, alpha=0.6, zorder=2)
        # Rotated along the rule itself: a horizontal note here landed on the curves in one
        # panel and under the legend in the other, and the panels cannot share a free corner.
        ax.annotate(text["truncation"].format(n=dropped), xy=(complete[-1], 0.5),
                    xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", fontsize=7.5, color="#555555",
                    rotation=90, va="center", ha="right")
        ax.set_xticks(sizes)
        ax.set_xlabel(text["xlabel"], fontsize=9)
        style(ax)

    left.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.suptitle(text["title"], x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def present(data: list[dict], size: int) -> list[float]:
    return [float(r[f"regret_n{size}"]) for r in data if r[f"regret_n{size}"]]


def gaps(data: list[dict], size: int) -> list[float]:
    """The statistical component per cell: what this cell pays beyond its own floor."""
    return [float(r[f"regret_n{size}"]) - float(r["transfer_only"]) for r in data
            if r[f"regret_n{size}"] and r["transfer_only"]]


FIGURES = [("front_position_vs_bound.png", figure_front_position),
           ("regret_vs_race_size.png", figure_race_size),
           ("quantum_sweep.png", figure_quantum),
           ("basin_vs_aco.png", figure_basins)]


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, _build in FIGURES:
        # Deleted before drawing, not overwritten after: a build that fails halfway must
        # not leave the previous run's picture behind wearing today's date.
        (FIGS / name).unlink(missing_ok=True)
    for name, build in FIGURES:
        build(FIGS / name)
        print(f"[done] {(FIGS / name).relative_to(ROOT)}")
    stale = sorted(p.name for p in FIGS.glob("*.png")
                   if p.name not in {n for n, _ in FIGURES})
    if stale:
        print(f"[note] {len(stale)} image(s) in figures/ are not produced by this script "
              f"and may predate the current results: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
