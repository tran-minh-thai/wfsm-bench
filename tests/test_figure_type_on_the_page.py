"""Every drawn figure uses an accepted, embedded font, and its text reaches the page at 9-10pt.

Read from the PDF files themselves, not from the builder's font-size arguments. The builder
sets sizes on a canvas that the typesetter then scales to the width the manuscript asks for,
and tight bounding boxes crop that canvas by an amount only the saved file knows -- so the
only size worth checking is the one in the file, multiplied by that scale.

What the check reads, and why each part is there:

* the font of every text-showing operator: the family must be Times New Roman, the font
  program must be TrueType (Type 3, matplotlib's default, is refused by the journal) and it
  must be embedded;
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
FIGS = ROOT / "results" / "figures"
ACCEPTED_FAMILY = "TimesNewRoman"
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
    low, high = 9.0, 10.0
    problems = []
    for font in {tuple(run["font"].items()) for run in runs}:
        font = dict(font)
        if ACCEPTED_FAMILY not in font["name"]:
            problems.append(f"{path.name}: font {font['name']} is not {ACCEPTED_FAMILY}")
        if font["subtype"] != "CIDFontType2" and font["subtype"] != "TrueType":
            problems.append(f"{path.name}: font {font['name']} is {font['subtype']}, not TrueType")
        if not font["embedded"]:
            problems.append(f"{path.name}: font {font['name']} is not embedded")
    largest_in_block = {}
    for run in runs:
        largest_in_block[run["block"]] = max(largest_in_block.get(run["block"], 0), run["size"])
    for run in runs:
        printed = run["size"] * scale
        is_index = run["size"] < largest_in_block[run["block"]] - 1e-6
        if is_index:
            if printed < INDEX_FLOOR_RATIO * low - TOLERANCE_PT:
                problems.append(f"{path.name}: a sub/superscript prints at {printed:.2f}pt")
        elif not (low - TOLERANCE_PT <= printed <= high + TOLERANCE_PT):
            problems.append(f"{path.name}: text prints at {printed:.2f}pt, outside {low}-{high}pt")
    return problems


# ---------------------------------------------------------------- the checks
def _drawn() -> list[tuple[Path, float]]:
    module = _builder()
    return [(FIGS / name, module.printed_width_in(FIGS / name)) for name, _ in module.FIGURES]


def test_every_figure_has_a_declared_print_width() -> None:
    module = _builder()
    assert set(module.PRINTED_FRACTION) == {name for name, _ in module.FIGURES}


@pytest.mark.parametrize("path,printed", _drawn(), ids=lambda value: getattr(value, "name", ""))
def test_figure_type_meets_the_journal_rules(path: Path, printed: float) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} is not drawn; run scripts/make_figures.py")
    width, runs = text_runs(path)
    problems = page_report(path, printed)
    # The denominator, so a pass over zero strings cannot read as a pass.
    assert len(runs) >= 10, f"{path.name}: only {len(runs)} strings read"
    assert not problems, "\n".join(problems[:20])


# ---------------------------------------------------------------- the check must be able to fail
def _draw(tmp_path: Path, family: str, fontsize: float, fonttype: int = 42) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with matplotlib.rc_context({"font.family": family, "pdf.fonttype": fonttype}):
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
    good = _draw(tmp_path, "Times New Roman", 9.5)
    assert page_report(good, 4.0) == []


@pytest.mark.parametrize("family,fontsize,fonttype,expected", [
    ("Times New Roman", 8.0, 42, "outside 9.0-10.0pt"),     # too small on the page
    ("Times New Roman", 11.0, 42, "outside 9.0-10.0pt"),    # too large on the page
    ("DejaVu Sans", 9.5, 42, "is not TimesNewRoman"),       # matplotlib's default family
    ("Times New Roman", 9.5, 3, "not TrueType"),            # matplotlib's default font type
])
def test_the_check_rejects_each_breach(tmp_path: Path, family, fontsize, fonttype,
                                       expected) -> None:
    bad = _draw(tmp_path, family, fontsize, fonttype)
    problems = page_report(bad, 4.0)
    assert any(expected in line for line in problems), problems


def test_the_check_reads_the_print_scale(tmp_path: Path) -> None:
    """The same file passes at the width it was drawn for and fails when shrunk to 0.86."""
    good = _draw(tmp_path, "Times New Roman", 9.5)
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
