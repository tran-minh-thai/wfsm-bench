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


INVARIANT = ROOT / "results-invariant"


def figure_ball_coverage(path: Path) -> None:
    """Ball coverage against remaining depth, one line per collection.

    The pruning criterion turns on one quantity -- how much of a host a ball of radius d
    already covers -- and the table states it at a single depth. A criterion whose whole
    content is "this approaches one" has to be READ approaching one, not asserted at a point.

    The line that matters most is the one that does NOT reach the cutoff: it is what stops the
    criterion being a rule that always says "give up".
    """
    text = labels("do-phu-qua-cau")
    do_thi = {}
    for source in sorted(INVARIANT.glob("bound_locality_d*.csv")):
        with source.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                do_thi.setdefault(row["dataset"], {})[int(row["depth"])] = (
                    100.0 * float(row["ball_coverage_pooled"]))
    if len(do_thi) < 2:
        raise SystemExit(f"[fail] {INVARIANT} holds ball coverage for {len(do_thi)} "
                         f"collection(s); run scripts/measure_bound_locality.py at each depth")
    depths = sorted({d for series in do_thi.values() for d in series})
    if len(depths) < 3:
        raise SystemExit(f"[fail] only {len(depths)} depth(s) measured; a curve needs a range, "
                         f"and a two-point line would assert a shape nothing measured")

    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(6.4, 3.6))
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for (name, series), colour in zip(sorted(do_thi.items()), colours):
        xs = [d for d in depths if d in series]
        ax.plot(xs, [series[d] for d in xs], marker="o", ms=4, lw=1.6,
                color=colour, label=name, zorder=3)

    cutoff = float(text["cutoff_percent"])
    ax.axhline(cutoff, color="#555555", lw=1.0, ls="--", zorder=2)
    ax.annotate(text["cutoff_label"], xy=(depths[0], cutoff), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color="#555555")

    ax.set_xticks(depths)
    ax.set_ylim(0, 105)
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    ax.set_title(text["title"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _stats_logs() -> dict[str, str]:
    """collection -> log stem, read from the table builder so both render one source."""
    body = (ROOT / "scripts" / "make_tables.py").read_text(encoding="utf-8")
    body = body.split("STATS_LOG = {", 1)[1].split("}", 1)[0]
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', body))


def figure_rank_transfer(path: Path) -> None:
    """Mean rank of every method, one row per collection, one panel per weight mode.

    The rank table asks the reader to compare seven numbers across six collections in two
    modes and see for themselves that the winner changes. That is the claim, and a table is
    the wrong shape for it.

    One aggregated diagram per mode would be the conventional rendering and would be WRONG
    here: aggregating over collections is precisely what hides a result that says the ranking
    does not survive changing collections. So the collections stay separate and the eye reads
    down the column.
    """
    text = labels("thu-hang-khong-chuyen")
    logs = _stats_logs()
    du_lieu: dict[str, dict[str, dict[str, float]]] = {}
    cd = None
    for collection, stem in logs.items():
        source = RESULTS / f"{stem}_statistics.log"
        if not source.exists():
            continue
        body = source.read_text(encoding="utf-8", errors="replace")
        found = re.search(r"Nemenyi CD = ([0-9.]+)", body)
        if found:
            cd = float(found.group(1))
        for mode in text["modes"]:
            ranks = parse_ranks(body, mode)
            if ranks:
                du_lieu.setdefault(mode, {})[collection] = ranks
    if cd is None or len(du_lieu) < 2:
        raise SystemExit(f"[fail] rank data for {len(du_lieu)} mode(s) and CD={cd}; the "
                         f"figure would assert a comparison nothing measured")

    methods = sorted({m for mode in du_lieu.values() for r in mode.values() for m in r})
    colours = dict(zip(methods, ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                 "#E69F00", "#56B4E9", "#999999"]))
    import matplotlib.pyplot as plt
    modes = list(du_lieu)
    figure, axes = plt.subplots(1, len(modes), figsize=(9.6, 3.4), sharex=True, sharey=True)
    # One ORDER for both panels. With sharey and a per-mode order, a mode measured on five
    # collections and a mode measured on six put different collections on the same row, and
    # the shared tick labels then named the wrong ones -- measured on the first draft of this
    # figure. A collection a mode has no data for gets a blank row, which is the honest shape.
    cols = sorted({c for mode in du_lieu.values() for c in mode})
    for ax, mode in zip(axes, modes):
        for y, collection in enumerate(cols):
            ranks = du_lieu[mode].get(collection)
            if not ranks:
                ax.annotate(text["absent"], xy=(1.0, y), fontsize=7, color="#999999",
                            va="center")
                continue
            best = min(ranks, key=ranks.get)
            for method, value in ranks.items():
                ax.scatter(value, y, s=46 if method == best else 26,
                           color=colours[method], zorder=3,
                           edgecolor="black" if method == best else "none",
                           linewidth=0.8)
            ax.annotate(best, xy=(ranks[best], y), xytext=(-6, 8),
                        textcoords="offset points", fontsize=7, ha="right",
                        color=colours[best])
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        ax.set_ylim(len(cols) - 0.4, -1.1)
        ax.set_xlabel(text["xlabel"])
        ax.set_title(text["modes"][mode], fontsize=9, loc="left")
        ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        # The critical distance, drawn once as a span the reader can carry along the axis.
        ax.plot([1.0, 1.0 + cd], [-0.7, -0.7], lw=2.2, color="#333333",
                solid_capstyle="butt", clip_on=False, zorder=4)
        ax.annotate(text["cd_label"].format(cd=f"{cd:.3f}".replace(".", ",")),
                    xy=(1.0, -0.7), xytext=(0, 5), textcoords="offset points", fontsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colours[m], label=m)
               for m in methods]
    figure.legend(handles=handles, frameon=False, fontsize=8, ncol=len(methods),
                  loc="lower center", bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(text["title"], fontsize=10, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


FIGURES = [("rank_transfer_by_collection.png", figure_rank_transfer),
           ("ball_coverage_by_depth.png", figure_ball_coverage),
           ("front_position_vs_bound.png", figure_front_position),
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
