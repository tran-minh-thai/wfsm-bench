"""Shared conditions for checks that reach outside this repository.

The manuscript, its label files and its rule files live in ../paper/, a sibling directory
that is deliberately NOT part of this repository: the code is published and the paper is not.
A machine that clones only the code therefore has none of them, and every check that reads
one must skip rather than fail. The measurement machine found seven such failures at once,
all of them saying the same thing in different words.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper"


def manuscript_file(name: str) -> Path:
    return PAPER / name


def needs_manuscript(*names: str):
    """Skip unless every named file beside the manuscript is present."""
    missing = [n for n in names if not (PAPER / n).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"outside this repository: {', '.join(missing)} (expected in {PAPER})")
