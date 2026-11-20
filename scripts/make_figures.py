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

# Every word drawn inside a figure comes from one label file in this repository, so a clone
# can redraw every figure and the words can be changed without touching the drawing code.
LABELS_PATH = ROOT / "figures" / "labels.en.json"


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
    return _escape_for_latex(loaded[figure])


def _escape_for_latex(value):
    """Label text goes through LaTeX (text.usetex): a bare % would start a comment and drop the
    rest of the string. Applied to every string of a figure's labels, nested ones included."""
    if isinstance(value, str):
        # | < > in OT1 text mode print as an em dash and inverted punctuation ("|P(K)|" came out
        # as "--P(K)--"), so they are set in math mode, as the body sets them.
        value = re.sub(r"(?<!\\)%", r"\\%", value)
        return re.sub(r"[|<>]", lambda m: f"${m.group(0)}$", value)
    if isinstance(value, dict):
        return {k: _escape_for_latex(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_escape_for_latex(v) for v in value]
    return value

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
NO_DATE_METADATA = {"CreationDate": None}

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

# Figure text is typeset by LaTeX in the manuscript's own face, Computer Modern (2026-11-20, the
# author: figure lettering "as close as possible to the body text", in face and size). The
# Springer class sets the body in Computer Modern at 10pt; matplotlib's own Computer Modern font
# has no U+2265 and the OpenType version embeds as CFF with viewer warnings, so the text goes
# through latex instead (text.usetex) and the PDF carries the same Type 1 fonts as the body.
# The journal recommends Helvetica or Arial for lettering; the author chose consistency with the
# body. Before, the figures used Arial (earlier on 2026-11-20) and Times New Roman (Elsevier).
# underscore: dataset and method names keep their underscores; newunicodechar: the few symbols
# the labels and builders print.
FIGURE_FONT = "Computer Modern"
# One design size for every size: a canvas drawn at 17pt and printed at 10pt would otherwise
# take the 17pt design (CMR17, measured 2026-11-20), visibly lighter than the body's CMR10.
LATEX_PREAMBLE = ("\\DeclareFontShape{OT1}{cmr}{m}{n}{<->cmr10}{}"
                  "\\DeclareFontShape{OML}{cmm}{m}{it}{<->cmmi10}{}"
                  "\\DeclareFontShape{OMS}{cmsy}{m}{n}{<->cmsy10}{}"
                  "\\usepackage{type1cm}\\usepackage{underscore}\\usepackage{newunicodechar}"
                  "\\newunicodechar{\u2192}{\\ensuremath{\\rightarrow}}"
                  "\\newunicodechar{\u00b7}{\\ensuremath{\\cdot}}"
                  "\\newunicodechar{\u2265}{\\ensuremath{\\geq}}"
                  "\\newunicodechar{\u2264}{\\ensuremath{\\leq}}"
                  "\\newunicodechar{\u00d7}{\\ensuremath{\\times}}"
                  "\\newunicodechar{\u2212}{\\ensuremath{-}}"
                  "\\newunicodechar{\u2013}{--}")
matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
                            "font.serif": ["Computer Modern Roman"],
                            "text.latex.preamble": LATEX_PREAMBLE})
# Text is measured on the raster canvas -- to place labels and to refuse overlaps -- and the
# canvas hints glyphs to its pixel grid. At 100 dpi that made every string about 5% narrower
# than the same string in the saved PDF (measured 2026-09-29: 159.8pt against 168.2pt of ink),
# so a 3pt gap on the canvas was a 7pt overlap on the page. Unhinted, the measured width is
# within 1% of the ink in the PDF. The output is vector, so nothing drawn changes.
matplotlib.rcParams["text.hinting"] = "no_hinting"

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


# The manuscript's text column: \linewidth of RaceSel.tex under the Springer Nature class,
# measured by building it on 2026-11-20 -- 372.0pt, 5.147in. (Under the Elsevier class it was
# 468.33pt.) Within the journal's limit of 174mm for a single-column figure.
COLUMN_WIDTH_IN = 372.0 / 72.27
# Each builder draws on a canvas whose width was chosen for the Elsevier column (6.48in) and is
# scaled to the printed width. The canvas narrows by the same factor as the column, so a figure
# prints at the scale it was tuned at: the type fits in the same 9-10pt and the printed HEIGHTS
# the author set stay as they were. Drawing the old canvases into the narrower column instead
# scaled them by 0.79, and the type no longer settled in 9-10pt (measured 2026-11-20).
ELSEVIER_COLUMN_IN = 6.48
NARROW = COLUMN_WIDTH_IN / ELSEVIER_COLUMN_IN

# Every string on the printed page lies in this range, in TeX points: one size for every figure,
# close to the 10pt body (the author, 2026-11-20). 9.5pt rather than 10: drawn at 10pt, the
# two-panel figures ran their panel titles and axis labels into each other in the 372pt column
# (measured 2026-11-20). Within the journal's 8-12pt. Before: 9-10pt, which left most figures at
# 9pt and a few strings at 10pt.
# Before 2026-09-29 sizes were chosen per figure, and three figures were never scaled at all:
# the basin labels, drawn at 7pt and printed at 0.86 of a 7.2in canvas, reached the page at
# about 5.4pt -- below Elsevier's 7pt floor for ordinary text.
PRINTED_PT = (9.45, 9.55)

# 2026-11-20, the author: printed heights reduced by these amounts (cm on the page), per figure;
# the front-position figure loses its amount on EACH row. Applied where each builder sets its
# height, converted to canvas inches through the scale at which the figure prints.
HEIGHT_CUT_CM = {"front_position_vs_bound.pdf": 0.5,
                 "quantum_sweep.pdf": 2.0, "regret_vs_race_size.pdf": 2.0,
                 "ball_coverage_by_depth.pdf": 3.0, "enumeration_cost_by_bound.pdf": 1.0}
CM = 1 / 2.54

# The share of the text column each figure is included at -- the width= of its
# \includegraphics in the manuscript. The text sizes are fitted to it, so a figure included
# at another width prints at another size; tests/test_figure_type_on_the_page.py reads the
# drawn files against this table.
PRINTED_FRACTION = {"enumeration_cost_by_bound.pdf": 1.0,
                    "ball_coverage_by_depth.pdf": 1.0,
                    "front_position_vs_bound.pdf": 0.86,
                    "regret_vs_race_size.pdf": 0.86,
                    "quantum_sweep.pdf": 0.86,
                    "basin_vs_aco.pdf": 1.0}   # full width since the Springer column (2026-11-20)


def printed_width_in(path: Path) -> float:
    return PRINTED_FRACTION[path.name] * COLUMN_WIDTH_IN


def require_font() -> None:
    """Refuse to draw when LaTeX is missing: without it matplotlib cannot set the figure text in
    the body face, and silently drawing in DejaVu instead is what this check exists to stop."""
    import shutil
    missing = [tool for tool in ("latex", "dvipng") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"[fail] {', '.join(missing)} not on PATH; the figure text is typeset "
                         f"by LaTeX in Computer Modern and cannot be drawn without it")

def _saved_width_in(fig, tight: bool) -> float:
    """The width the saved PDF will have: the tight bounding box plus its padding, or the
    canvas when the figure is saved without cropping."""
    if not tight:
        return fig.get_size_inches()[0]
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    return box.width + 2 * matplotlib.rcParams["savefig.pad_inches"]


def _refuse_overlapping_text(fig) -> None:
    """Stop when two strings of the laid-out figure cover each other.

    Every size check can pass on a figure whose labels are printed across one another --
    enlarging the type to the printed range is exactly what makes that happen -- and it is
    only seen by opening the file. This measures it instead, on the final layout.
    """
    import matplotlib.text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Tick labels are kept by matplotlib for ticks outside the view limits and simply not
    # drawn; their stale positions would report clashes nobody can see. Only the ticks the
    # axis draws are measured.
    all_tick_labels, drawn_tick_labels = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_tick_labels.update((id(tick.label1), id(tick.label2)))
            for tick in axis._update_ticks():
                drawn_tick_labels.update((id(tick.label1), id(tick.label2)))
    boxes = []
    for text_artist in fig.findobj(matplotlib.text.Text):
        if id(text_artist) in all_tick_labels and id(text_artist) not in drawn_tick_labels:
            continue
        if text_artist.get_visible() and text_artist.get_text().strip():
            box = text_artist.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                boxes.append((text_artist.get_text(), box))
    # Not merely disjoint: two strings closer than a point read as touching on the page.
    gap = 1.0 * fig.dpi / 72.0
    boxes = [(label, box.padded(gap / 2)) for label, box in boxes]
    clashes = [(a, b) for i, (a, box_a) in enumerate(boxes) for b, box_b in boxes[i + 1:]
               if box_a.overlaps(box_b)]
    if clashes:
        raise SystemExit(f"[fail] {fig.get_label() or 'a figure'}: {len(clashes)} pair(s) of "
                         f"strings overlap on the laid-out figure, first "
                         f"{clashes[0][0]!r} / {clashes[0][1]!r}")


def _fit_text_to_print(fig, printed_width_in: float, rect=None, tight: bool = True,
                       **layout) -> None:
    """Size every string so that it reaches the page between PRINTED_PT[0] and PRINTED_PT[1].

    The ratio is taken against the width of the file that will be SAVED, not the canvas: the
    figures are saved with a tight bounding box, which crops or extends the canvas, and the
    typesetter scales the saved width to the width the manuscript asks for. Resizing the text
    moves that bounding box, so the fit is repeated until it holds. The layout is re-run with
    the rectangle the builder used each time; without that the figure keeps room reserved for
    the old type size, which once put an axis label across its own tick numbers.
    """
    import matplotlib.text
    low, high = PRINTED_PT
    for _ in range(8):
        fig.tight_layout(**layout) if rect is None else fig.tight_layout(rect=rect, **layout)
        ratio = printed_width_in / _saved_width_in(fig, tight)
        out_of_range = 0
        for text_artist in fig.findobj(matplotlib.text.Text):
            if not text_artist.get_text().strip():
                continue
            printed = text_artist.get_fontsize() * ratio
            if low <= printed <= high:
                continue
            out_of_range += 1
            # A hundredth of a point inside each end: the saved width is measured from the
            # drawn canvas and the file's own box differs from it in the last digits, which
            # once put a label at 10.01pt.
            text_artist.set_fontsize(min(max(printed, low + 0.01), high - 0.01) / ratio)
        if not out_of_range:
            _refuse_overlapping_text(fig)
            return
    raise SystemExit(f"[fail] the text of {fig.get_label() or 'a figure'} did not settle "
                     f"inside {low}-{high}pt on the page after eight passes")


def figure_quantum(path: Path) -> None:
    """Plateau structure varies twentyfold; no method's rank follows it."""
    text = labels("dia-hinh-bi-ep-doi")
    levels, ranks = [], []
    for tag in QUANTA:
        land = rows(f"weight_mode_landscape_{tag}.csv")
        levels.append(float(land[0]["fitness_levels"]))
        got = mean_ranks(f"{tag}_statistics.log", "sum")
        ranks.append(got)

    fig, ax = plt.subplots(figsize=(7.2 * NARROW, 4.4 - HEIGHT_CUT_CM[path.name] * CM * 7.2 * NARROW
                                    / printed_width_in(path)))
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
    # No figure title: the journal asks for none inside an illustration; the caption carries it.
    # Anchored a fixed distance in points below the axes, not a fraction of the axes height:
    # a fraction shrinks with the figure, and at the shorter height of 2026-11-20 it put the
    # legend on top of the axis label.
    from matplotlib.transforms import offset_copy
    ax.legend(frameon=False, fontsize=10, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              bbox_transform=offset_copy(ax.transAxes, fig=fig, y=-44, units="points"))
    style(ax)
    _fit_text_to_print(fig, printed_width_in(path))
    fig.savefig(path, dpi=200, metadata=NO_DATE_METADATA, bbox_inches="tight")
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

    # Drawn at the width it prints at, so that marks, labels and gaps measured on the canvas
    # are the ones on the page up to the small crop of the tight bounding box.
    MARK_AREA = 70                                 # scatter size, pt^2
    fig, ax = plt.subplots(figsize=(printed_width_in(path), printed_width_in(path) * 0.7))
    text = labels("luu-vuc-va-dan-kien")
    # Two groups, told apart by colour, marker AND fill: solid diamonds against hollow circles.
    for dominates, colour, mark, label in ((True, "#D55E00", "D", text["legend_dominates"]),
                                           (False, "#0072B2", "o", text["legend_not"])):
        group = [p for p in points if p[2] is dominates]
        ax.scatter([p[0] for p in group], [p[1] for p in group], s=MARK_AREA, marker=mark,
                   facecolors=colour if dominates else "white", edgecolors=colour if not dominates
                   else "#1A1A1A", linewidths=1.6 if not dominates else 0.8,
                   label=label, zorder=3)
    # The scale is set HERE, before anything reads a display coordinate. It used to be set
    # further down, after this placement had already measured the figure on a linear x axis
    # -- a layout that never reached the page, so every clearance test was answering about a
    # different picture.
    ax.set_xscale("log")
    # Room at both ends for a name beside the outermost points. With matplotlib's default
    # margin the leftmost and rightmost names could only go inward, onto their neighbours.
    # The right end stops at 100: the axis is a share of the pool, and an axis running to
    # 280% invites a reader to ask what lives there.
    shares = [p[0] for p in points]
    ax.set_xlim(min(shares) / 4, min(max(shares) * 4, 100.0))
    # The reference line and its words share one colour of their own (the author, 2026-11-20), a
    # dark green, unlike the orange and blue of the two point groups; dashed and heavier
    # than a grid line, so it also stands apart in greyscale.
    RANK_ONE_COLOUR = "#1B5E20"   # dark green (the author, 2026-11-20)
    ax.axhline(1.0, color=RANK_ONE_COLOUR, linewidth=1.2, linestyle="--", zorder=1)
    # The one string on this canvas that was still English in the printed draft: every
    # other word here came from the label file and this one was typed in place.
    # Above the line, at its right end (the author, 2026-11-20: under it the axis cut the
    # letters). The point names are placed afterwards with this label as an obstacle.
    rank_one = ax.annotate(text["rank_one"], xy=(ax.get_xlim()[1], 1.0), xytext=(0, 2),
                           textcoords="offset points", color=RANK_ONE_COLOUR, va="bottom",
                           ha="right")
    legend = ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    # The title states the half that holds without exception. The converse -- a
    # non-dominant optimum meaning ACO wins -- holds in six of eight, so claiming the
    # figure shows basin structure "tracking" the ranking would overstate it.
    # No figure title: the journal asks for none inside an illustration; the caption carries it.
    style(ax)
    # Laid out and sized BEFORE the point names are placed, and the names are then kept out
    # of the layout: placing them first and laying out afterwards moved every mark, and on
    # the 2026-09-29 drawing two names cleared at placement were printed across each other.
    _fit_text_to_print(fig, printed_width_in(path))
    ratio = printed_width_in(path) / _saved_width_in(fig, tight=True)
    LABEL_FONTSIZE = (PRINTED_PT[0] + 0.1) / ratio
    renderer = fig.canvas.get_renderer()

    # Where each label goes is decided against the marks, not by alternating sides. The
    # alternating version put every label to the RIGHT and only varied up or down, so a point
    # with a neighbour just right of it had its name written across that neighbour: measured
    # on the drawn figure, 2 of 13 marks carried label pixels inside them.
    px_per_pt = fig.dpi / 72.0
    MARK_RADIUS = (MARK_AREA ** 0.5 / 2 + 0.8) * px_per_pt     # drawn radius with its edge
    # Two labels that merely do not overlap can still read as one phrase ("BZR·sum QM9·disc"
    # on the 2026-09-29 drawing), so neighbours keep this many pixels between them.
    LABEL_GAP = 3 * px_per_pt
    obstacles = [tuple(artist.get_window_extent(renderer).extents) for artist in (legend, rank_one)]
    mark_positions = [ax.transData.transform((b, r)) for b, r, *_ in sorted(points)]
    axes_box = ax.get_window_extent(renderer)
    ordered = sorted(points)

    def name_of(point) -> str:
        dataset, mode = point[3], point[4]
        return f"{dataset.replace('_bond', '').replace('_full', '')}·{mode[:4]}"

    def candidate_boxes(index: int) -> list[tuple[float, float, tuple]]:
        # Measured, not estimated: the estimate (0.62 em a character) was sized for DejaVu
        # Sans and overstates Times by a third, which rejected positions that were clear.
        probe = ax.text(0, 0, name_of(ordered[index]), fontsize=LABEL_FONTSIZE)
        probe_box = probe.get_window_extent(renderer)
        probe.remove()
        w, h = probe_box.width, probe_box.height
        x, y = mark_positions[index]
        # The offset clears the mark's own radius: starting closer put the first letter
        # inside the very mark the label names.
        off = MARK_RADIUS + 2 * px_per_pt
        offsets = [(off, 2), (off, -2 - h), (-off - w, 2), (-off - w, -2 - h),
                   (-w / 2, MARK_RADIUS + 2), (-w / 2, -MARK_RADIUS - 2 - h),
                   (off, h + 4), (off, -2 * h - 4), (-off - w, h + 4), (-off - w, -2 * h - 4)]
        return [(dx, dy, (x + dx, y + dy, x + dx + w, y + dy + h)) for dx, dy in offsets]

    def hits_marks(index: int, box) -> int:
        # Every mark EXCEPT this label's own: sitting beside its own point is the
        # arrangement, not a collision, and testing against it rejected all four corners.
        return sum(not (box[2] < mx - MARK_RADIUS or box[0] > mx + MARK_RADIUS
                        or box[3] < my - MARK_RADIUS or box[1] > my + MARK_RADIUS)
                   for j, (mx, my) in enumerate(mark_positions) if j != index)

    def leaves_axes(box) -> bool:
        # Inside the axes: the leftmost label, placed to the left of its point, was
        # written across the y axis and its tick labels.
        return (box[0] < axes_box.x0 or box[2] > axes_box.x1
                or box[1] < axes_box.y0 or box[3] > axes_box.y1)

    # The most constrained names are placed first. In the order of the x axis the rightmost
    # name, hemmed in by the axis end, came last and found its few free positions already
    # taken by its neighbours' names.
    freedom = {index: sum(not leaves_axes(box) and not hits_marks(index, box)
                          for _dx, _dy, box in candidate_boxes(index))
               for index in range(len(ordered))}
    placed_boxes = []
    for index in sorted(range(len(ordered)), key=lambda i: (freedom[i], i)):
        basin, rank = ordered[index][0], ordered[index][1]
        label_text = name_of(ordered[index])
        scored = []
        for order, (dx, dy, label_box) in enumerate(candidate_boxes(index)):
            # Across as well as along: the overlap test that runs on the finished figure wants
            # a point between any two strings, and a name placed flush under another passed
            # this test and failed that one.
            hits_label = sum(not (label_box[2] + LABEL_GAP < o[0] or label_box[0] - LABEL_GAP > o[2]
                                  or label_box[3] + px_per_pt < o[1] or label_box[1] - px_per_pt > o[3])
                             for o in placed_boxes + obstacles)
            scored.append((1000 * leaves_axes(label_box) + 10 * hits_label
                           + hits_marks(index, label_box), order, dx, dy, label_box))
        # The first clear position in the order above; when none is clear, the one that
        # collides least, and it is SAID -- a label printed over another is a figure fault
        # nobody sees until the page is read. Before 2026-09-29 the fallback was a fixed
        # corner, which on the printed-size drawing put two names across each other.
        score, _order, dx, dy, label_box = min(scored)
        if score:
            print(f"[warn] basin label {label_text!r} has no clear position (score {score})")
        placed_boxes.append(label_box)
        name = ax.annotate(label_text, (basin, rank), fontsize=LABEL_FONTSIZE, color="#444444",
                           xytext=(dx / px_per_pt, dy / px_per_pt),
                           textcoords="offset points", ha="left", va="bottom")
        name.set_in_layout(False)
    # The names are inside the axes, so the layout and the saved width do not move; the
    # sizes and the overlap test are run again on the finished figure.
    _fit_text_to_print(fig, printed_width_in(path))
    fig.savefig(path, dpi=200, metadata=NO_DATE_METADATA, bbox_inches="tight")
    plt.close(fig)

SELECTOR_ORDER = ["weighted", "ideal", "maxmin"]
SELECTOR_COLOUR = {"weighted": "#D55E00", "ideal": "#0072B2", "maxmin": "#009E73"}
# 2026-11-20, the author: every series is told apart by colour AND line style AND marker (AND
# fill where an area is drawn), so a figure survives a greyscale print and a reader who cannot
# separate the hues. One fixed identity per selector and per dataset, used in every figure.
SELECTOR_MARKER = {"weighted": "o", "ideal": "s", "maxmin": "^"}
SELECTOR_DASH = {"weighted": "-", "ideal": "--", "maxmin": "-."}
# Datasets: the editing rules' palette (D.9), no two hues of one colour; red is kept free for a
# proposed method. Each dataset also owns a marker and a dash.
DATASET_STYLE = {
    "Cuneiform":         ("#1F77B4", "o", "-"),
    "QM9_bond":          ("#FF7F0E", "s", "--"),
    "BZR_bond":          ("#2CA02C", "^", "-."),
    "COX2_bond":         ("#9467BD", "D", ":"),
    "DHFR_bond":         ("#8C564B", "v", (0, (5, 1, 1, 1, 1, 1))),
    "alchemy_full_bond": ("#000000", "P", (0, (6, 2))),
}


def dataset_style(name: str) -> tuple[str, str, object]:
    if name not in DATASET_STYLE:
        raise SystemExit(f"[fail] no style for dataset {name!r}; add it to DATASET_STYLE so it "
                         f"is not drawn in a default that another dataset may share")
    return DATASET_STYLE[name]


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
    row_in = 3.2 - HEIGHT_CUT_CM[path.name] * CM * 10.4 * NARROW / printed_width_in(path)
    fig, axes = plt.subplots(rows_of, columns, figsize=(10.4 * NARROW, row_in * rows_of),
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
            mark = SELECTOR_MARKER[selector]
            for bound in bounds:
                values = per_bound.get(bound, [])
                ax.scatter([bound] * len(values), values, s=14, color=colour, marker=mark,
                           alpha=0.30, zorder=2, linewidth=0)
            xs = [b for b in bounds if per_bound.get(b)]
            ax.plot(xs, [statistics.mean(per_bound[b]) for b in xs], marker=mark,
                    markersize=5, linewidth=1.8, color=colour, zorder=3,
                    linestyle=SELECTOR_DASH[selector],
                    label=text["selectors"][selector] if index == 0 else None)
        ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_title(dataset, loc="left", fontsize=9)
        ax.set_xticks(bounds)
        ax.set_ylim(-0.08, 1.12)
        style(ax)
    for spare in flat[len(names):]:
        spare.set_visible(False)

    # The two reference lines are named where they END, right of the top-right panel. Named
    # inside the first panel, at the printed size, "front maximum" sat on that panel's own
    # points at the first bound and "front floor" hung below the axis across the tick marks.
    last_in_row = flat[min(columns, len(names)) - 1]
    for key, level in (("top_line", 1.0), ("bottom_line", 0.0)):
        last_in_row.annotate(text[key], xy=(1.0, level), xycoords=("axes fraction", "data"),
                             xytext=(4, 0), textcoords="offset points", color="#555555",
                             va="center", ha="left", annotation_clip=False)
    fig.supxlabel(text["xlabel"], fontsize=9)
    fig.supylabel(text["ylabel"], fontsize=9)
    # No figure title: the journal asks for none inside an illustration; the caption carries it.
    # Two columns and more room under the panels: on the Springer column three entries on one
    # row met the shared axis label (2026-11-20).
    # One row (the author, 2026-11-20), with compact handles and gaps so the row fits the column.
    fig.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", handlelength=1.6,
               handletextpad=0.4, columnspacing=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    _fit_text_to_print(fig, printed_width_in(path), rect=(0.02, 0.03, 1, 1))
    fig.savefig(path, dpi=200, metadata=NO_DATE_METADATA, bbox_inches="tight")
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
    print_scale = printed_width_in(path) / (10.4 * NARROW)
    fig, axes = plt.subplots(1, 2, figsize=(10.4 * NARROW, (2.35 + 1.15 - HEIGHT_CUT_CM[path.name] * CM)
                                            / print_scale))

    # Left: the mean total regret against the mean floor. Averaging is linear, so the gap
    # between the two lines IS the mean of the per-cell gaps -- no cross-cell subtraction.
    left = axes[0]
    full_y = [statistics.mean(present(data, n)) for n in complete]
    bal_y = [statistics.mean([float(r[f"regret_n{n}"]) for r in balanced]) for n in sizes]
    full_floor = statistics.mean([float(r["transfer_only"]) for r in data if r["transfer_only"]])
    bal_floor = statistics.mean([float(r["transfer_only"]) for r in balanced])
    # An OPAQUE light tint (the orange at 13% on white) under an opaque hatch. A face with an
    # alpha makes the PDF backend apply that alpha to the hatch lines too, and the hatch then
    # vanishes on the page.
    left.fill_between(complete, full_y, full_floor, facecolor="#F9EADE", hatch="////",
                      hatchcolor="#D55E00", edgecolor="none", zorder=1, label=text["gap"])
    left.plot(complete, full_y, marker="o", markersize=5, linewidth=2.0, color="#D55E00",
              zorder=4, label=text["full"].format(n=len(data)))
    left.axhline(full_floor, color="#D55E00", linewidth=1.2, linestyle="--", zorder=3,
                 label=text["floor_full"].format(n=len(data)))
    left.plot(sizes, bal_y, marker="s", markersize=4, linewidth=1.6, color="#0072B2",
              linestyle="-.", zorder=4, label=text["balanced"].format(n=len(balanced)))
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
               linewidth=1.6, color="#0072B2", linestyle="-.", zorder=4,
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
                              edgecolor="none", alpha=0.7),
                    # Above the curves (drawn at 4), or the plate is painted underneath them
                    # and the lines run across the letters at full strength.
                    zorder=5)
        ax.set_xticks(sizes)
        ax.set_xlabel(text["xlabel"], fontsize=9)
        style(ax)

    # The legend sits OUTSIDE the axes, under both panels. Inside, it was drawn in the upper
    # right of a box too short to hold it, so it printed across the tick marks and the axis
    # label it was sitting on top of. It describes both panels anyway, so the figure is the
    # right owner of it, not the left panel.
    handles, label_text = left.get_legend_handles_labels()
    # Two columns: three were wider than the Springer column (2026-11-20).
    # Compact handles and gaps: at body size in Computer Modern the two columns were wider than
    # the canvas, and the fit kept widening the saved box (2026-11-20).
    fig.legend(handles, label_text, frameon=False, fontsize=7.5, ncol=2, handlelength=1.6,
               handletextpad=0.5, columnspacing=1.0,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    # No figure title: the journal asks for none inside an illustration; the caption carries it.
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    _fit_text_to_print(fig, printed_width_in(path), rect=(0, 0.15, 1, 1))
    fig.savefig(path, dpi=200, metadata=NO_DATE_METADATA, bbox_inches="tight")
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

    No threshold line is drawn (removed 2026-11-20): the criterion fixes no value of tau, and a
    dashed line at one value would state a threshold the text does not. The line that matters
    is the collection that stays far from one: it is what stops the criterion being a rule that
    always says "give up".
    """
    text = labels("do-phu-qua-cau")
    coverage_by_collection = {}
    for source in sorted(INVARIANT.glob("bound_locality_d*.csv")):
        with source.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                coverage_by_collection.setdefault(row["dataset"], {})[int(row["depth"])] = (
                    100.0 * float(row["ball_coverage_pooled"]))
    if len(coverage_by_collection) < 2:
        raise SystemExit(f"[fail] {INVARIANT} holds ball coverage for {len(coverage_by_collection)} "
                         f"collection(s); run scripts/measure_bound_locality.py at each depth")
    depths = sorted({d for series in coverage_by_collection.values() for d in series})
    if len(depths) < 3:
        raise SystemExit(f"[fail] only {len(depths)} depth(s) measured; a curve needs a range, "
                         f"and a two-point line would assert a shape nothing measured")

    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(6.4 * NARROW, 3.6 - HEIGHT_CUT_CM[path.name] * CM * 6.4 * NARROW
                                       / printed_width_in(path)))
    for name, series in sorted(coverage_by_collection.items()):
        colour, mark, dash = dataset_style(name)
        xs = [d for d in depths if d in series]
        ax.plot(xs, [series[d] for d in xs], marker=mark, ms=4, lw=1.6, linestyle=dash,
                color=colour, label=name, zorder=3)

    ax.set_xticks(depths)
    ax.set_ylim(0, 105)
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    # No figure title: the journal asks for none inside an illustration; the caption carries it.
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    _fit_text_to_print(figure, printed_width_in(path), tight=False)
    figure.savefig(path, dpi=200, metadata=NO_DATE_METADATA)
    plt.close(figure)


def _stats_logs() -> dict[str, str]:
    """collection -> log stem, read from the table builder so both render one source."""
    body = (ROOT / "scripts" / "make_tables.py").read_text(encoding="utf-8")
    body = body.split("STATS_LOG = {", 1)[1].split("}", 1)[0]
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', body))


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
                earlier_cell = data.setdefault(row["dataset"], {}).get(bound)
                # The same cell in two artefact files is not a duplicate to be deduplicated:
                # it is two measurements of one thing, and this figure cannot choose between
                # them. Cuneiform bound 6 sits in two files for exactly this reason -- one
                # timed both sides, the other timed gSpan alone -- and only the timed one
                # reaches here, because the other leaves exhaustive_seconds empty.
                if earlier_cell is not None and earlier_cell != cell:
                    raise SystemExit(
                        f"[fail] {row['dataset']} bound {bound} is measured twice and the two "
                        f"disagree: pool {earlier_cell[0]} vs {cell[0]}, {earlier_cell[1]:.4f} vs {cell[1]:.4f} "
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
    too_few = sorted(f"{name} ({len(cells)})" for name, cells in data.items() if len(cells) < 3)
    if too_few:
        raise SystemExit(
            f"[fail] {len(too_few)} collection(s) have fewer than three size bounds measured: "
            f"{', '.join(too_few)}. Two points make a straight line on any axis, so the curve "
            f"would assert a growth rate nothing measured. Run stage 20.")

    import matplotlib.pyplot as plt
    # Same arithmetic as the two figures above. Drawn 10.0in wide, printed at COT_IN, so
    # the canvas shrinks to 0.648 of itself; at the old height the left panel's axis label
    # ran into the right panel's, and the left panel's y label into the suptitle. The height
    # is chosen so the printed panels are as tall as they are wide is not the point -- the
    # point is that it is chosen in printed inches at all.
    print_scale = printed_width_in(path) / (10.0 * NARROW)
    figure, axes = plt.subplots(1, 2, figsize=(10.0 * NARROW, (2.55 + 1.05 - HEIGHT_CUT_CM[path.name] * CM)
                                              / print_scale),
                               sharex=True)
    bounds = sorted({k for cells in data.values() for k, _pool, _per in cells})

    for name, cells in sorted(data.items()):
        colour, mark, dash = dataset_style(name)
        xs = [k for k, _pool, _per in cells]
        axes[0].plot(xs, [pool for _k, pool, _per in cells], marker=mark, ms=4.5, lw=1.7,
                     linestyle=dash, color=colour, label=name, zorder=3)
        axes[1].plot(xs, [per for _k, _pool, per in cells], marker=mark, ms=4.5, lw=1.7,
                     linestyle=dash, color=colour, label=name, zorder=3)

    for ax, ylabel, panel in ((axes[0], text["ylabel_pool"], text["panel_pool"]),
                              (axes[1], text["ylabel_per"], text["panel_per"])):
        ax.set_yscale("log")
        ax.set_xticks(bounds)
        ax.set_ylabel(ylabel, fontsize=9)
        # Wrapped: on one line the right panel's title ran past the canvas in the 372pt column,
        # and the fit kept widening the saved box (2026-11-20).
        ax.set_title("\n".join(textwrap.wrap(panel, 26)), loc="left", fontsize=9)
        style(ax)
        # The y grid is what makes a slope comparable between two lines that sit in different
        # decades, and on a log axis the minor lines are the ones that carry the reading.
        ax.grid(axis="y", which="minor", alpha=0.12, linewidth=0.5)

    handles = [plt.Line2D([], [], marker=dataset_style(name)[1], ms=4.5, lw=1.7,
                          linestyle=dataset_style(name)[2], color=dataset_style(name)[0],
                          label=name)
               for name, _cells in sorted(data.items())]
    # Two columns with compact handles: three were wider than the canvas at body size (2026-11-20).
    # Two columns with compact handles: one row of six measures 451pt at 9.5pt against the 372pt
    # column (2026-11-20), and three columns were wider than the canvas too.
    figure.legend(handles=handles, frameon=False, fontsize=8, ncol=2, handlelength=1.6,
                  handletextpad=0.5, columnspacing=1.0,
                  loc="lower center", bbox_to_anchor=(0.5, -0.075))
    # One x label for both panels, not one each. The panels share the axis (sharex), so the
    # second copy said nothing the first had not; it was also the copy that overhung the right
    # edge, because it is centred under an axes whose right side IS the figure's right side,
    # and the crop then took its closing bracket off.
    figure.supxlabel(text["xlabel"], fontsize=9, y=0.105)
    # No figure title: the journal asks for none inside an illustration; the caption carries it.
    # w_pad separates the two panels: the left panel's x label is long enough to reach under
    # the right panel's y label, and tight_layout's default gutter was measured for the type
    # size before the scaling pass, not after it.
    figure.tight_layout(rect=(0, 0.11, 1, 1), w_pad=2.4)
    _fit_text_to_print(figure, printed_width_in(path), rect=(0, 0.11, 1, 1), w_pad=2.4)
    figure.savefig(path, dpi=200, metadata=NO_DATE_METADATA, bbox_inches="tight")
    plt.close(figure)


# The rank-transfer figure was removed on 2026-11-20: the manuscript keeps the rank table, which
# shows the same mean ranks with the precision the critical-difference comparisons need.
FIGURES = [("enumeration_cost_by_bound.pdf", figure_enumeration_cost),
           ("ball_coverage_by_depth.pdf", figure_ball_coverage),
           ("front_position_vs_bound.pdf", figure_front_position),
           ("regret_vs_race_size.pdf", figure_race_size),
           ("quantum_sweep.pdf", figure_quantum),
           ("basin_vs_aco.pdf", figure_basins)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Draw every figure from the tracked artefacts.")
    ap.add_argument("--out", type=Path, default=FIGS,
                    help="directory to write the figures to (default: results/figures)")
    out = ap.parse_args(argv).out
    require_font()
    if set(PRINTED_FRACTION) != {name for name, _ in FIGURES}:
        raise SystemExit(f"[fail] PRINTED_FRACTION and FIGURES name different figures: "
                         f"{sorted(set(PRINTED_FRACTION) ^ {name for name, _ in FIGURES})}")
    out.mkdir(parents=True, exist_ok=True)
    for name, _build in FIGURES:
        # Deleted before drawing, not overwritten after: a build that fails halfway must
        # not leave the previous run's picture behind wearing today's date.
        (out / name).unlink(missing_ok=True)
    for name, build in FIGURES:
        build(out / name)
        print(f"[done] {(out / name).as_posix()}")
    # Every file, not only one extension. The figures moved from PNG to PDF on 2026-09-13, and
    # a scan that looked for PNGs would have reported the directory clean while seven stale
    # rasters sat in it -- which is the exact failure this note exists to report.
    stale = sorted(p.name for p in out.iterdir()
                   if p.is_file() and not p.name.startswith(".")
                   and p.name not in {n for n, _ in FIGURES})
    if stale:
        print(f"[note] {len(stale)} image(s) in figures/ are not produced by this script "
              f"and may predate the current results: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
