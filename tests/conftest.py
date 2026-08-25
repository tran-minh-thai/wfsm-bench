"""Conditions for the few checks here that reach outside this repository.

The manuscript and its label files live in a sibling directory that is deliberately not part
of this repository: the code is published and the paper is not. A machine that clones only
the code has none of them, so a check that reads one must skip rather than fail.

Most manuscript checking has moved to the paper side. What is left here are checks about the
CODE that happen to need a label file to run -- the figure builder, for instance, cannot draw
without its axis text.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper"


def needs_manuscript(*names: str):
    missing = [n for n in names if not (PAPER / n).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"outside this repository: {', '.join(missing)} (expected in {PAPER})")
