"""Every drawn figure uses the body face, embedded, and its text reaches the page at body size.

Read from the PDF files themselves, not from the builder's font-size arguments. The builder
sets sizes on a canvas that the typesetter then scales to the width the manuscript asks for,
and tight bounding boxes crop that canvas by an amount only the saved file knows -- so the
only size worth checking is the one in the file, multiplied by that scale.

What the check reads, and why each part is there:

* the font of every text-showing operator: the family must be Computer Modern (the body face of
  the Springer class, set by LaTeX through text.usetex; the author asked on 2026-11-20 for figure
  lettering close to the body text), the font must be an outline font (Type 3, matplotlib's
  default for its own fonts, is a bitmap-like procedure font) and it must be embedded;
* the effective size of every string: the Tf operand times the scale of the text matrix and
  the current transformation matrix -- rotation does not change it, so rotated axis labels are
  checked like the rest;
* sub- and superscripts, recognised by STRUCTURE (a smaller string in the same BT..ET block as
  a larger one: matplotlib's "10^1" tick labels), not by a size ratio -- a 7pt note beside a
  10pt label has exactly the superscript ratio and is still ordinary text.
"""
from __future__ import annotations

import importlib.util
import math
import re
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"
# Computer Modern Type 1 fonts are named CMR10, CMMI10, CMSY10, CMBX10, ... after the subset tag.
ACCEPTED_FAMILY = "Computer Modern"
ACCEPTED_PREFIX = "CM"
# The journal's floor for sub- and superscripts is 6pt; matplotlib sets them at 0.7 of the
# base size, and the base is at least 9pt, so 0.7 * 9 is what a correct figure produces.
INDEX_FLOOR_RATIO = 0.7
TOLERANCE_PT = 0.02


def _builder():
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- a minimal PDF reader
_OBJECT = re.compile(rb"(\d+)\s+0\s+obj(.*?)endobj", re.S)
_TOKEN = re.compile(rb"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>]+|"
                    rb"[-+]?(?:\d+\.?\d*|\.\d+)|[A-Za-z'\"*]+")


def _objects(data: bytes) -> dict[int, bytes]:
    return {int(m.group(1)): m.group(2) for m in _OBJECT.finditer(data)}


def _stream(body: bytes) -> bytes | None:
    match = re.search(rb"stream\r?\n(.*)endstream", body, re.S)
    if match is None:
        return None
    raw = match.group(1)
    return zlib.decompress(raw) if b"/FlateDecode" in body else raw


def _ref(body: bytes, key: bytes) -> int | None:
    match = re.search(rb"/" + key + rb"\s+(\d+)\s+0\s+R", body)
    return int(match.group(1)) if match else None


def _fonts(objects: dict[int, bytes], resources: bytes) -> dict[str, dict]:
    """Resource name -> {name, subtype, embedded} for the fonts a page can use."""
    font_dict_ref = _ref(resources, b"Font")
    font_dict = objects[font_dict_ref] if font_dict_ref else resources
    found = {}
    for name, number in re.findall(rb"/([A-Za-z0-9_.+-]+)\s+(\d+)\s+0\s+R", font_dict):
        body = objects.get(int(number), b"")
        if b"/Type /Font" not in body and b"/Type/Font" not in body:
            continue
        subtype = re.search(rb"/Subtype\s*/(\w+)", body).group(1).decode()
        descriptor_body = body
        if subtype == "Type0":
            descendant = re.search(rb"/DescendantFonts\s*\[\s*(\d+)\s+0\s+R", body)
            descriptor_body = objects[int(descendant.group(1))]
            subtype = re.search(rb"/Subtype\s*/(\w+)", descriptor_body).group(1).decode()
        base = re.search(rb"/BaseFont\s*/([^\s/\[\]<>]+)", body)
        descriptor = _ref(descriptor_body, b"FontDescriptor")
        embedded = bool(descriptor and re.search(rb"/FontFile[23]?\s", objects[descriptor]))
        found[name.decode()] = {"name": base.group(1).decode() if base else "?",
                                "subtype": subtype, "embedded": embedded}
    return found


def _multiply(a, b):
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5])


def text_runs(path: Path) -> tuple[float, list[dict]]:
    """(page width in pt, every shown string with its font and effective size in canvas pt)."""
    data = path.read_bytes()
    objects = _objects(data)
    page = next(body for body in objects.values() if re.search(rb"/Type\s*/Page\b", body))
    width = float(re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+[-\d.]+\s+([-\d.]+)", page).group(2))
    resources = objects[_ref(page, b"Resources")] if _ref(page, b"Resources") else page
    fonts = _fonts(objects, resources)
    contents = re.search(rb"/Contents\s*(\d+)\s+0\s+R", page)
    stream = _stream(objects[int(contents.group(1))])

    runs, stack, operands = [], [], []
    ctm = (1, 0, 0, 1, 0, 0)
    tm = (1, 0, 0, 1, 0, 0)
    font, size, block = None, 0.0, 0
    for token in _TOKEN.findall(stream):
        if token[:1] in b"(<[]/" or re.fullmatch(rb"[-+]?(?:\d+\.?\d*|\.\d+)", token):
            operands.append(token)
            continue
        op = token
        numbers = []
        for item in operands:
            try:
                numbers.append(float(item))
            except ValueError:
                pass
        if op == b"q":
            stack.append(ctm)
        elif op == b"Q":
            ctm = stack.pop()
        elif op == b"cm" and len(numbers) >= 6:
            ctm = _multiply(tuple(numbers[-6:]), ctm)
        elif op == b"BT":
            tm, block = (1, 0, 0, 1, 0, 0), block + 1
        elif op == b"Tm" and len(numbers) >= 6:
            tm = tuple(numbers[-6:])
        elif op == b"Tf":
            font, size = operands[-2][1:].decode(), numbers[-1]
        elif op in (b"Tj", b"TJ", b"'", b'"'):
            matrix = _multiply(tm, ctm)
            scale = math.sqrt(abs(matrix[0] * matrix[3] - matrix[1] * matrix[2]))
            shown = b"".join(item for item in operands if item[:1] in b"(<")
            if shown.strip(b"()<> "):
                runs.append({"font": fonts.get(font, {"name": f"<undeclared {font}>",
                                                      "subtype": "?", "embedded": False}),
                             "size": size * scale, "block": block})
        operands = []
    return width, runs


def page_report(path: Path, printed_width_in: float) -> list[str]:
    """Every breach of the font and size rules in one file, as readable lines."""
    width, runs = text_runs(path)
    if not runs:
        return [f"{path.name}: no text found -- the reader, not the figure, is broken"]
    scale = printed_width_in * 72.0 / width
    low, high = _printed_range()
    problems = []
    for font in {tuple(run["font"].items()) for run in runs}:
        font = dict(font)
        if not font["name"].split("+")[-1].startswith(ACCEPTED_PREFIX):
            problems.append(f"{path.name}: font {font['name']} is not {ACCEPTED_FAMILY}")
        if font["subtype"] == "Type3":
            problems.append(f"{path.name}: font {font['name']} is Type3, not an outline font")
        if not font["embedded"]:
            problems.append(f"{path.name}: font {font['name']} is not embedded")
    largest_in_block = {}
    for run in runs:
        largest_in_block[run["block"]] = max(largest_in_block.get(run["block"], 0), run["size"])
    for run in runs:
        # In TeX points: the PDF states sizes in PostScript points (72 to the inch), while the
        # range is the body size, which LaTeX states in TeX points (72.27 to the inch). Text set
        # by LaTeX at 9.91pt reads 9.87 in the PDF; compared unconverted it failed the range.
        printed = run["size"] * scale * 72.27 / 72.0
        is_index = run["size"] < largest_in_block[run["block"]] - 1e-6
        if is_index:
            if printed < INDEX_FLOOR_RATIO * low - TOLERANCE_PT:
                problems.append(f"{path.name}: a sub/superscript prints at {printed:.2f}pt")
        elif not (low - TOLERANCE_PT <= printed <= high + TOLERANCE_PT):
            problems.append(f"{path.name}: text prints at {printed:.2f}pt, outside {low}-{high}pt")
    return problems


# ---------------------------------------------------------------- the checks
def _drawn() -> list[tuple[str, float]]:
    module = _builder()
    return [(name, module.printed_width_in(Path(name))) for name, _ in module.FIGURES]


def test_every_figure_has_a_declared_print_width() -> None:
    module = _builder()
    assert set(module.PRINTED_FRACTION) == {name for name, _ in module.FIGURES}


@pytest.mark.parametrize("name,printed", _drawn(), ids=lambda value: value if isinstance(value, str) else "")
def test_figure_type_meets_the_journal_rules(name: str, printed: float, drawn_figures) -> None:
    path = drawn_figures / name
    assert path.exists(), f"{name} was not drawn by scripts/make_figures.py"
    width, runs = text_runs(path)
    problems = page_report(path, printed)
    # The denominator, so a pass over zero strings cannot read as a pass.
    assert len(runs) >= 10, f"{path.name}: only {len(runs)} strings read"
    assert not problems, "\n".join(problems[:20])


# ---------------------------------------------------------------- the check must be able to fail
def _printed_range() -> tuple[float, float]:
    """The builder's own range, so the check and the drawing cannot drift apart."""
    global _RANGE
    try:
        return _RANGE
    except NameError:
        _RANGE = tuple(_builder().PRINTED_PT)
        return _RANGE


def _draw(tmp_path: Path, family: str, fontsize: float, fonttype: int = 42) -> Path:
    """family "latex" draws through text.usetex in Computer Modern, as the builder does."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = ({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]}
          if family == "latex" else {"text.usetex": False, "font.family": family,
                                     "pdf.fonttype": fonttype})
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot([1, 10, 100], [1, 2, 3])
        ax.set_xscale("log")
        ax.set_xlabel("depth", fontsize=fontsize)
        ax.set_ylabel("coverage", fontsize=fontsize)
        ax.set_title("a title", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        out = tmp_path / f"{family.replace(' ', '')}_{fontsize}_{fonttype}.pdf"
        fig.savefig(out)                    # no cropping: the page is exactly 4in wide
        plt.close(fig)
    return out


def test_a_correct_figure_passes(tmp_path: Path) -> None:
    good = _draw(tmp_path, "latex", sum(_printed_range()) / 2)
    assert page_report(good, 4.0) == []


@pytest.mark.parametrize("family,fontsize,fonttype,expected", [
    ("latex", 9.0, 42, "outside"),                          # the old 9pt, below body size
    ("latex", 11.0, 42, "outside"),                         # too large on the page
    ("DejaVu Sans", 9.95, 42, "is not Computer Modern"),    # matplotlib's default family
    ("Arial", 9.95, 42, "is not Computer Modern"),          # the face used earlier on 2026-11-20
    ("DejaVu Sans", 9.95, 3, "not an outline font"),        # matplotlib's default font type
])
def test_the_check_rejects_each_breach(tmp_path: Path, family, fontsize, fonttype,
                                       expected) -> None:
    bad = _draw(tmp_path, family, fontsize, fonttype)
    problems = page_report(bad, 4.0)
    assert any(expected in line for line in problems), problems


def test_the_check_reads_the_print_scale(tmp_path: Path) -> None:
    """The same file passes at the width it was drawn for and fails when shrunk to 0.86."""
    good = _draw(tmp_path, "latex", sum(_printed_range()) / 2)
    assert page_report(good, 4.0) == []
    assert any("outside" in line for line in page_report(good, 0.86 * 4.0))


def test_the_builder_refuses_overlapping_strings() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    module = _builder()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.text(0.5, 0.5, "one label")
    ax.text(0.52, 0.5, "another label")
    with pytest.raises(SystemExit, match="overlap"):
        module._refuse_overlapping_text(fig)
    plt.close(fig)


def test_the_overlap_check_ignores_ticks_outside_the_view() -> None:
    """matplotlib keeps tick labels for ticks beyond the limits and does not draw them. Their
    stale positions once reported '-0.25' over '1.25' on a figure where neither is printed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axis import Axis
    module = _builder()

    def two_close_panels():
        # Ticks at -0.25 and 1.25 exist but lie outside the limits, and the panels are close
        # enough that the lower panel's hidden 1.25 sits where the upper panel's labels are.
        fig, axes = plt.subplots(2, 1, figsize=(4.0, 4.0), sharex=True, sharey=True)
        for ax in axes:
            ax.plot([0, 1], [0, 1])
            ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25])
            ax.set_ylim(-0.08, 1.12)
            ax.tick_params(labelsize=12)
        fig.subplots_adjust(hspace=0.08)
        return fig

    module._refuse_overlapping_text(two_close_panels())       # must not raise
    # The case is not vacuous: counting every tick as drawn reports the stale clash.
    original = Axis._update_ticks
    try:
        Axis._update_ticks = lambda self: (original(self),
                                           self.get_major_ticks() + self.get_minor_ticks())[1]
        with pytest.raises(SystemExit, match="overlap"):
            module._refuse_overlapping_text(two_close_panels())
    finally:
        Axis._update_ticks = original
    plt.close("all")
