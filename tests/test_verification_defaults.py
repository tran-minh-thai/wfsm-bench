"""The constants the theory checks repeat must be the ones the study ran with.

verification.py repeats the study defaults on purpose, so a change of default forces the
checks to be re-derived. The repetition drifted once: the size reference was copied from the
unit-test default (6) instead of the study anchor (7) that every recorded fitness uses.
"""

from __future__ import annotations

from wfsmbench.config import STUDY_SIZE_REFERENCE, Params
from wfsmbench.measure import verification as V


def test_the_size_reference_is_the_study_anchor() -> None:
    assert V.SIZE_REFERENCE == STUDY_SIZE_REFERENCE


def test_the_objective_weights_are_the_parameter_defaults() -> None:
    p = Params()
    assert (V.ALPHA, V.BETA, V.GAMMA, V.ETA1, V.ETA2) == (p.alpha, p.beta, p.gamma, p.eta1, p.eta2)
