"""The same dataset gets the same checksum on Windows and macOS.

On 2026-10-16 a fresh clone rebuilt the eight datasets on macOS and seven of them hashed
differently from the measurement machine's record although their content was identical: the
converters wrote JSON in text mode, which is CRLF on Windows, and the record hashed raw bytes.
Two guards follow. The converters write LF on every system, and the record hashes the content
with line endings normalised, which also covers files already written with CRLF.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITERS = ROOT / "src" / "wfsmbench" / "io"


def _environment():
    spec = importlib.util.spec_from_file_location("environment", ROOT / "scripts" / "environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crlf_and_lf_copies_of_one_file_hash_alike() -> None:
    env = _environment()
    lf = b'{\n  "graphs": [1, 2]\n}\n'
    assert env.content_hash(lf.replace(b"\n", b"\r\n")) == env.content_hash(lf)
    assert env.content_hash(lf) == hashlib.sha256(lf).hexdigest()
    assert env.content_hash(lf) != env.content_hash(lf.replace(b"2", b"3"))


def text_writes_without_newline(source: str) -> list[int]:
    """Lines of .write_text(...) or open(..., "w"...) calls that leave line endings to the OS."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute | ast.Name):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        keywords = {k.arg for k in node.keywords}
        if name == "write_text" and "newline" not in keywords:
            found.append(node.lineno)
        if name == "open" and "newline" not in keywords:
            modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
            modes += [k.value.value for k in node.keywords
                      if k.arg == "mode" and isinstance(k.value, ast.Constant)]
            if any("w" in m and "b" not in m for m in modes):
                found.append(node.lineno)
    return found


def test_the_check_catches_each_form() -> None:
    assert text_writes_without_newline('p.write_text(s, encoding="utf-8")\n') == [1]
    assert text_writes_without_newline('open(p, "w", encoding="utf-8")\n') == [1]
    assert text_writes_without_newline('Path(p).open(mode="w")\n') == [1]
    assert text_writes_without_newline('p.write_text(s, encoding="utf-8", newline="\\n")\n') == []
    assert text_writes_without_newline('open(p, "wb")\n') == []
    assert text_writes_without_newline('open(p, encoding="utf-8")\n') == []


def test_every_dataset_writer_fixes_its_line_endings() -> None:
    files = sorted(WRITERS.glob("*.py"))
    writers = [f for f in files if "write_text" in f.read_text(encoding="utf-8")]
    assert len(writers) >= 3, f"only {len(writers)} writer module(s) found: {writers}"
    offending = {f.name: lines for f in files
                 if (lines := text_writes_without_newline(f.read_text(encoding="utf-8")))}
    assert not offending, f"text writes that would be CRLF on Windows: {offending}"
