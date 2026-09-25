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
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))

from wfsmbench.measure.ranks import mean_ranks as parse_ranks  # noqa: E402
FIGS = RESULTS / "figures"

# Axis labels, titles and legends are ENGLISH, and the draft they go into is Vietnamese. That
# is deliberate, decided 2026-09-13: a figure drawn once in English survives the translation
# pass, while a Vietnamese one would have to be redrawn from the artefacts and re-checked at
# exactly the point where nobody wants to be re-checking figures. The caption under each
# figure stays in the draft and stays Vietnamese; only what is inside the picture is here.
#
# The words still live beside the manuscript rather than in this file, for the reason they
# always did: this file is published source and stays ASCII, and the labels are prose that
# the manuscript side edits.
LABELS_PATH = ROOT.parent / "paper" / "figure_labels.en.json"


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

# Vector, not raster. A PDF figure sets no resolution for the typesetter to fight with, prints
# at whatever the page gives it, and keeps its labels as TEXT -- so a reader can select a
# collection name out of the paper, and a later translation pass can be checked by reading the
# file rather than by looking at pixels.
#
# fonttype 42 embeds the glyphs as TrueType rather than tracing them as paths. Paths would
# still LOOK right and would lose exactly what the format was chosen for: the text.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 6

# No creation date inside the file. Without this every rebuild writes seven different files
# for the same measurements, the copier reports all seven stale on every run, and the check
# that exists to say "this figure no longer matches its artefact" says it constantly -- which
# is the same as saying nothing. Passed at save time because there is no rcParam for it.
KHONG_NGAY = {"CreationDate": None}

# A dark rim on every marker, for every figure in this file. Measured 2026-09-12 on a print
# threshold: a marker filled with its series colour has the luminance of that colour and
# nothing else, so the three palest members of the palette lost their markers along with
# their lines and left a legend entry pointing at blank paper. The rim sits below the
# threshold whatever the fill is, so the shape survives even where the colour does not.
#
# A default rather than an argument at each call: the palette is shared across figures, and a
# per-figure fix is one a new figure does not inherit.
matplotlib.rcParams["lines.markeredgecolor"] = "#1A1A1A"
matplotlib.rcParams["lines.markeredgewidth"] = 0.7

# Nemenyi critical difference for seven methods over eighty blocks. Drawn on the sweep
# so a reader can see at a glance that nothing moves by a distinguishable amount.
CD = 1.007
METHODS = ["aco", "pso", "multiswarm", "genetic", "clpso", "local_search", "random"]
# Colourblind-safe (Okabe-Ito), fixed order so a rebuild never recolours a line.
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000", "#999999"]
# A second channel for the same seven, because the paper is read on paper. Hue alone is not an
# encoding once a page is photocopied: at print threshold the three palest colours above go to
# white and their series leave the figure without anything saying they are gone. Shape and dash
# do not depend on the reproduction, so each method keeps its identity in a greyscale proof.
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
DASHES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
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


# The draft's text column, measured from the one figure that is neither shrunk nor stretched:
# ball coverage is drawn 6.4in wide, asked for at \linewidth, and its 10pt label arrives at
# 10.13pt -- so the column is 6.48in. A figure asked for at 0.86\linewidth prints at 5.57in.
COT_IN = 6.48


def _phong_chu(fig, rong_in: float, rect=None) -> None:
    """Scale every string in the figure by the inverse of the shrink the draft will apply.

    Called before saving, so the numbers the builders pass stay readable as what the PAGE
    receives. Not applied to figures already printing at their nominal size: scaling their
    text would be a change with no reader on the other end of it.

    The layout is then re-run with the rectangle the builder used. Without that the figure
    keeps a layout computed for the smaller type -- measured on the first build of this
    change, the front-position figure's axis label ran straight across its own tick numbers,
    because room had been reserved for 4.9pt text and 8pt text was put into it.
    """
    import matplotlib.text
    he_so = rong_in / fig.get_size_inches()[0]
    if he_so >= 0.95:
        return
    for chu in fig.findobj(matplotlib.text.Text):
        chu.set_fontsize(chu.get_fontsize() / he_so)
    fig.tight_layout() if rect is None else fig.tight_layout(rect=rect)


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
    for method, colour, mark, dash in zip(METHODS, COLOURS, MARKERS, DASHES):
        ax.plot(levels, [r[method] for r in ranks], marker=mark, markersize=5,
                linestyle=dash,
                linewidth=1.6, color=colour, label=method)
    # The critical difference drawn against the flattest interesting line, so the
    # comparison a reader needs -- movement against the threshold of significance -- is
    # on the page rather than left to arithmetic.
    base = ranks[0]["local_search"]
    ax.axhspan(base - CD / 2, base + CD / 2, color="#999999", alpha=0.13, zorder=0)
    ax.annotate(text["cd"], xy=(levels[-1], base + CD / 2), fontsize=10,
                color="#555555", va="bottom", ha="right")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    ax.set_title(text["title"], loc="left")
    ax.legend(frameon=False, fontsize=10, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, metadata=KHONG_NGAY, bbox_inches="tight")
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
    # Where each label goes is decided against the marks, not by alternating sides. The
    # alternating version put every label to the RIGHT and only varied up or down, so a point
    # with a neighbour just right of it had its name written across that neighbour: measured
    # on the drawn figure, 2 of 13 marks carried label pixels inside them.
    CO_CHU, CAO_CHU, BAN_KINH = 7, 7, 13.0   # fontsize, line height, mark radius in px
    # The scale is set HERE, before anything reads a display coordinate. It used to be set
    # further down, after this placement had already measured the figure on a linear x axis
    # -- a layout that never reached the page, so every clearance test was answering about a
    # different picture.
    ax.set_xscale("log")
    fig.canvas.draw()                        # transData is only valid once laid out
    vi_tri = [ax.transData.transform((b, r)) for b, r, *_ in sorted(points)]
    da_dat = []
    for index, (basin, rank, _flag, dataset, mode) in enumerate(sorted(points)):
        ten = f"{dataset.replace('_bond', '').replace('_full', '')}·{mode[:4]}"
        # Width from the string: no renderer measurement is needed for a monospaced-enough
        # estimate at this size, and over-estimating is the safe direction for a clearance
        # test.
        rong = 0.62 * CO_CHU * len(ten) * fig.dpi / 72.0
        cao = CAO_CHU * fig.dpi / 72.0
        x, y = vi_tri[index]
        chon = None
        # The offset clears the mark's own radius: starting closer put the first letter
        # inside the very mark the label names.
        xa = BAN_KINH + 3
        for dx, dy, ha in ((xa, 3, "left"), (xa, -3 - cao, "left"),
                           (-xa - rong, 3, "right"), (-xa - rong, -3 - cao, "right")):
            hop = (x + dx, y + dy, x + dx + rong, y + dy + cao)
            # Every mark EXCEPT this label's own: sitting beside its own point is the
            # arrangement, not a collision, and testing against it rejected all four corners.
            va_dau = any(not (hop[2] < mx - BAN_KINH or hop[0] > mx + BAN_KINH
                              or hop[3] < my - BAN_KINH or hop[1] > my + BAN_KINH)
                         for j, (mx, my) in enumerate(vi_tri) if j != index)
            va_chu = any(not (hop[2] < o[0] or hop[0] > o[2]
                              or hop[3] < o[1] or hop[1] > o[3]) for o in da_dat)
            if not (va_dau or va_chu):
                chon = (dx, dy, ha, hop)
                break
        if chon is None:
            # Nothing clear: keep the old placement rather than drop the label. A name in a
            # crowded corner is still readable; a missing name is a point nobody can identify.
            chon = (BAN_KINH + 3, 3 if index % 2 == 0 else -3 - cao, "left", None)
        dx, dy, ha, hop = chon
        if hop is not None:
            da_dat.append(hop)
        ax.annotate(ten, (basin, rank), fontsize=CO_CHU, color="#444444",
                    xytext=(dx * 72.0 / fig.dpi, dy * 72.0 / fig.dpi),
                    textcoords="offset points", ha="left", va="bottom")
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    # The one string on this canvas that was still English in the printed draft: every
    # other word here came from the label file and this one was typed in place.
    ax.annotate(text["rank_one"], xy=(ax.get_xlim()[1], 1.0), fontsize=8,
                color="#555555",
                va="bottom", ha="right")
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
    fig.savefig(path, dpi=200, metadata=KHONG_NGAY, bbox_inches="tight")
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
    _phong_chu(fig, 0.86 * COT_IN, rect=(0.02, 0.06, 1, 0.95))
    fig.savefig(path, dpi=200, metadata=KHONG_NGAY, bbox_inches="tight")
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

    # Drawn 10.4in wide and printed at 0.86 of the column, so every length on the canvas
    # arrives at 0.536 of itself. At the old height the printed axes box came out shorter than
    # three lines of its own tick labels, which is not an axes box, and the legend and the
    # truncation note were then drawn into a space that no longer existed. The height is set
    # from the PRINTED box the panels need, not from a canvas nobody sees.
    CO_CHU_IN = 0.86 * COT_IN / 10.4
    fig, axes = plt.subplots(1, 2, figsize=(10.4, (2.35 + 1.15) / CO_CHU_IN))

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
        # Wrapped, and wrapped HERE rather than in the label file: the words are the
        # manuscript's, the number of characters that fit beside a rule is this figure's.
        # Unwrapped, the line ran longer than the axes box is tall and crossed the title
        # above it and the axis label below -- a note about the figure printed over the
        # figure. Wrapping turns its length into width, which the panel has to spare.
        ax.annotate("\n".join(textwrap.wrap(text["truncation"].format(n=dropped), 24)),
                    xy=(complete[-1], 0.5),
                    xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", fontsize=7, color="#555555",
                    rotation=90, va="center", ha="right", ma="center",
                    # The note stands inside the data area because the rule it explains does.
                    # A pale plate keeps it legible where it crosses a curve without hiding
                    # the curve: the reader can still see the line run under the words.
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.7))
        ax.set_xticks(sizes)
        ax.set_xlabel(text["xlabel"], fontsize=9)
        style(ax)

    # The legend sits OUTSIDE the axes, under both panels. Inside, it was drawn in the upper
    # right of a box too short to hold it, so it printed across the tick marks and the axis
    # label it was sitting on top of. It describes both panels anyway, so the figure is the
    # right owner of it, not the left panel.
    handles, ten = left.get_legend_handles_labels()
    fig.legend(handles, ten, frameon=False, fontsize=7.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(text["title"], x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _phong_chu(fig, 0.86 * COT_IN, rect=(0, 0.10, 1, 0.94))
    fig.savefig(path, dpi=200, metadata=KHONG_NGAY, bbox_inches="tight")
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
    figure.savefig(path, dpi=200, metadata=KHONG_NGAY)
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
    marks = dict(zip(methods, MARKERS))
    colours = dict(zip(methods, ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                 "#E69F00", "#56B4E9", "#999999"]))
    import matplotlib.pyplot as plt
    modes = list(du_lieu)
    cols = sorted({c for mode in du_lieu.values() for c in mode})
    # Height is DERIVED from the row count, not fixed. The figure is drawn 9.6in wide and
    # printed at COT_IN, so every length on the canvas arrives at 0.675 of itself; a height
    # chosen once for a canvas the reader never sees is how six rows ended up closer together
    # than the height of one line of their own labels. Here the row pitch is stated in PRINTED
    # inches -- 0.30in against a 10pt label, which is a line and a half -- the fixed furniture
    # (both titles, the axis label, the legend, the critical-distance strip) is added in the
    # same units, and only then is the canvas height the quotient of the two.
    CO_CHU_IN = COT_IN / 9.6
    cao_in = (0.30 * len(cols) + 1.55) / CO_CHU_IN
    figure, axes = plt.subplots(1, len(modes), figsize=(9.6, cao_in),
                                sharex=True, sharey=True)
    # One ORDER for both panels. With sharey and a per-mode order, a mode measured on five
    # collections and a mode measured on six put different collections on the same row, and
    # the shared tick labels then named the wrong ones -- measured on the first draft of this
    # figure. A collection a mode has no data for gets a blank row, which is the honest shape.
    for ax, mode in zip(axes, modes):
        for y, collection in enumerate(cols):
            ranks = du_lieu[mode].get(collection)
            if not ranks:
                # Near-black, not the palette's grey. This line explains a row that carries
                # no mark at all, and in that grey it was the first thing a print threshold
                # removed -- leaving an empty row and nothing saying why, which reads as lost
                # data rather than as a cell that was never measured. It is text about the
                # figure, not one of the series, so it takes the colour of text.
                ax.annotate(text["absent"], xy=(1.0, y), fontsize=9, color="#1A1A1A",
                            va="center")
                continue
            best = min(ranks, key=ranks.get)
            for method, value in ranks.items():
                # scatter does not inherit the marker-edge default above, so it is stated
                # here; without it this figure keeps the defect the default exists to fix.
                # The rim is on EVERY mark now, not only the winner's. It used to be the
                # winner's badge, which left the other six with the luminance of their fill
                # and nothing else; the winner is still marked, by a thicker rim and a
                # larger mark.
                ax.scatter(value, y, s=52 if method == best else 30,
                           color=colours[method], marker=marks[method], zorder=3,
                           edgecolor="#1A1A1A",
                           linewidth=0.9 if method == best else 0.5)
            # The winner's NAME is not written here. The legend maps every symbol to its
            # method, so the word repeated what the mark already said, and it collided with
            # the neighbouring marks while doing it. The larger mark and thicker rim above
            # are what single the winner out.
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=10)
        # The critical-distance strip is BELOW the last row, not above the first. Above, it
        # shared the band with the panel title and the two printed on top of each other: the
        # title is drawn outside the axes and the strip inside it, so nothing in the layout
        # engine knew they were neighbours. Below, the strip has a band of its own and the
        # title has the whole band above the box.
        ax.set_ylim(len(cols) + 0.45, -0.6)
        ax.set_xlabel(text["xlabel"])
        ax.set_title(text["modes"][mode], fontsize=10, loc="left")
        ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        # The critical distance, drawn once as a span the reader can carry along the axis.
        ax.plot([1.0, 1.0 + cd], [len(cols) - 0.15, len(cols) - 0.15], lw=2.2,
                color="#333333", solid_capstyle="butt", clip_on=False, zorder=4)
        # A point, not a comma: the figure reads in English now, and a comma decimal inside an
        # English label is the one place the two languages would visibly collide.
        ax.annotate(text["cd_label"].format(cd=f"{cd:.3f}"),
                    xy=(1.0, len(cols) - 0.15), xytext=(0, 5),
                    textcoords="offset points", fontsize=8)
    handles = [plt.Line2D([], [], marker=marks[m], ls="", color=colours[m], label=m)
               for m in methods]
    figure.legend(handles=handles, frameon=False, fontsize=10, ncol=len(methods),
                  loc="lower center", bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(text["title"], fontsize=10, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    _phong_chu(figure, COT_IN, rect=(0, 0.06, 1, 0.94))
    figure.savefig(path, dpi=200, metadata=KHONG_NGAY, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------------ enumeration cost


def enumeration_cost() -> dict[str, list[tuple[int, int, float]]]:
    """collection -> [(size bound, pool size, seconds per pattern)], timed rows only.

    Read from every gspan_baseline_*.csv rather than from one of them: the timed cells are
    written per collection, and a file list kept here would go stale at the next collection.
    Rows without exhaustive_seconds are the gSpan-only cells -- they carry a pool size but no
    enumeration time, so a per-pattern cost computed from them would be a division by a
    number that was never measured.
    """
    data: dict[str, dict[int, tuple[int, float]]] = {}
    for source in sorted(RESULTS.glob("gspan_baseline_*.csv")):
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["exhaustive_seconds"]:
                    continue
                bound, pool = int(row["size_bound"]), int(row["exhaustive_patterns"])
                cell = (pool, float(row["exhaustive_seconds"]) / pool)
                cu = data.setdefault(row["dataset"], {}).get(bound)
                # The same cell in two artefact files is not a duplicate to be deduplicated:
                # it is two measurements of one thing, and this figure cannot choose between
                # them. Cuneiform bound 6 sits in two files for exactly this reason -- one
                # timed both sides, the other timed gSpan alone -- and only the timed one
                # reaches here, because the other leaves exhaustive_seconds empty.
                if cu is not None and cu != cell:
                    raise SystemExit(
                        f"[fail] {row['dataset']} bound {bound} is measured twice and the two "
                        f"disagree: pool {cu[0]} vs {cell[0]}, {cu[1]:.4f} vs {cell[1]:.4f} "
                        f"seconds per pattern. Two files claim the same cell; the figure "
                        f"cannot pick one.")
                data[row["dataset"]][bound] = cell
    return {name: [(k, *cells[k]) for k in sorted(cells)] for name, cells in data.items()}


def figure_enumeration_cost(path: Path) -> None:
    """Both factors of the enumeration cost against the size bound, on a log axis.

    The cost table prints both growth ratios already divided, which concedes that the reader
    needs the SHAPE and then asks them to join the points themselves. Straight lines on a log
    axis are what "geometric in K" looks like, and parallel lines across collections are what
    "the ratio is stable" looks like -- neither is readable from a column of ratios.

    Refuses under three bounds on any collection. Two points join into a line whatever the
    truth is, so a two-point panel would draw a growth rate nothing measured; that refusal is
    why this figure did not exist before the third bound was run on 2026-09-11.
    """
    text = labels("chi-phi-liet-ke")
    data = enumeration_cost()
    if not data:
        raise SystemExit("[fail] no timed enumeration rows in results/gspan_baseline_*.csv; "
                         "run stage 18 and stage 20 on the measurement machine")
    thieu = sorted(f"{name} ({len(cells)})" for name, cells in data.items() if len(cells) < 3)
    if thieu:
        raise SystemExit(
            f"[fail] {len(thieu)} collection(s) have fewer than three size bounds measured: "
            f"{', '.join(thieu)}. Two points make a straight line on any axis, so the curve "
            f"would assert a growth rate nothing measured. Run stage 20.")

    import matplotlib.pyplot as plt
    # Same arithmetic as the two figures above. Drawn 10.0in wide, printed at COT_IN, so
    # the canvas shrinks to 0.648 of itself; at the old height the left panel's axis label
    # ran into the right panel's, and the left panel's y label into the suptitle. The height
    # is chosen so the printed panels are as tall as they are wide is not the point -- the
    # point is that it is chosen in printed inches at all.
    CO_CHU_IN = COT_IN / 10.0
    figure, axes = plt.subplots(1, 2, figsize=(10.0, (2.55 + 1.05) / CO_CHU_IN),
                               sharex=True)
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    bounds = sorted({k for cells in data.values() for k, _pool, _per in cells})

    for (name, cells), colour in zip(sorted(data.items()), colours):
        xs = [k for k, _pool, _per in cells]
        axes[0].plot(xs, [pool for _k, pool, _per in cells], marker="o", ms=4.5, lw=1.7,
                     color=colour, label=name, zorder=3)
        axes[1].plot(xs, [per for _k, _pool, per in cells], marker="o", ms=4.5, lw=1.7,
                     color=colour, label=name, zorder=3)

    for ax, ylabel, panel in ((axes[0], text["ylabel_pool"], text["panel_pool"]),
                              (axes[1], text["ylabel_per"], text["panel_per"])):
        ax.set_yscale("log")
        ax.set_xticks(bounds)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(panel, loc="left", fontsize=9)
        style(ax)
        # The y grid is what makes a slope comparable between two lines that sit in different
        # decades, and on a log axis the minor lines are the ones that carry the reading.
        ax.grid(axis="y", which="minor", alpha=0.12, linewidth=0.5)

    handles = [plt.Line2D([], [], marker="o", ms=4.5, lw=1.7, color=colour, label=name)
               for (name, _cells), colour in zip(sorted(data.items()), colours)]
    figure.legend(handles=handles, frameon=False, fontsize=8, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, -0.075))
    # One x label for both panels, not one each. The panels share the axis (sharex), so the
    # second copy said nothing the first had not; it was also the copy that overhung the right
    # edge, because it is centred under an axes whose right side IS the figure's right side,
    # and the crop then took its closing bracket off.
    figure.supxlabel(text["xlabel"], fontsize=9, y=0.105)
    figure.suptitle(text["title"], fontsize=10, x=0.01, ha="left")
    # w_pad separates the two panels: the left panel's x label is long enough to reach under
    # the right panel's y label, and tight_layout's default gutter was measured for the type
    # size before the scaling pass, not after it.
    figure.tight_layout(rect=(0, 0.11, 1, 0.94), w_pad=2.4)
    _phong_chu(figure, COT_IN, rect=(0, 0.11, 1, 0.94))
    figure.savefig(path, dpi=200, metadata=KHONG_NGAY, bbox_inches="tight")
    plt.close(figure)


FIGURES = [("rank_transfer_by_collection.pdf", figure_rank_transfer),
           ("enumeration_cost_by_bound.pdf", figure_enumeration_cost),
           ("ball_coverage_by_depth.pdf", figure_ball_coverage),
           ("front_position_vs_bound.pdf", figure_front_position),
           ("regret_vs_race_size.pdf", figure_race_size),
           ("quantum_sweep.pdf", figure_quantum),
           ("basin_vs_aco.pdf", figure_basins)]


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, _build in FIGURES:
        # Deleted before drawing, not overwritten after: a build that fails halfway must
        # not leave the previous run's picture behind wearing today's date.
        (FIGS / name).unlink(missing_ok=True)
    for name, build in FIGURES:
        build(FIGS / name)
        print(f"[done] {(FIGS / name).relative_to(ROOT)}")
    # Every file, not only one extension. The figures moved from PNG to PDF on 2026-09-13, and
    # a scan that looked for PNGs would have reported the directory clean while seven stale
    # rasters sat in it -- which is the exact failure this note exists to report.
    stale = sorted(p.name for p in FIGS.iterdir()
                   if p.is_file() and not p.name.startswith(".")
                   and p.name not in {n for n, _ in FIGURES})
    if stale:
        print(f"[note] {len(stale)} image(s) in figures/ are not produced by this script "
              f"and may predate the current results: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
