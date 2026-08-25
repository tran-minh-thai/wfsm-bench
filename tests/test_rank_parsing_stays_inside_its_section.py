"""Mean ranks for one objective mode must come from that mode's section and no other.

The log holds one section per mode, each with its own mean-rank block. Two earlier copies of
this parse found the block by splitting the whole file at every occurrence of the heading
text and taking a segment between two of them. That is right only while the file holds
exactly the modes it happens to hold: give the tail more than one block and a dictionary
comprehension keeps the LAST name it sees, so the numbers reported for one mode become
another mode's.

It is a silent failure. Every value is a plausible mean rank, every method is present, and
the table renders. It was caught only because a regenerated artefact was compared against the
committed one before being committed over it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wfsmbench.measure.ranks import mean_ranks

RESULTS = Path(__file__).resolve().parents[1] / "results"

LOG = """\
header line

=== sum ===
Friedman over 80 blocks: chi2 = 1.0, p = 1e-9
mean ranks (1 = best), Nemenyi CD = 1.007:
    alpha        1.100
    beta         2.200

=== discounted ===
Friedman over 80 blocks: chi2 = 2.0, p = 1e-9
mean ranks (1 = best), Nemenyi CD = 1.007:
    alpha        7.700
    beta         8.800
"""


def test_each_mode_reads_its_own_numbers() -> None:
    assert mean_ranks(LOG, "sum") == {"alpha": 1.1, "beta": 2.2}
    assert mean_ranks(LOG, "discounted") == {"alpha": 7.7, "beta": 8.8}


def test_the_first_mode_does_not_absorb_the_second() -> None:
    """The exact failure: the earlier version returned 7.7 and 8.8 here."""
    first = mean_ranks(LOG, "sum")
    second = mean_ranks(LOG, "discounted")
    assert first != second, "both modes returned the same numbers; one section leaked"
    assert max(first.values()) < min(second.values()), (
        f"the first section returned {first}, which is the second section's data")


def test_a_mode_that_is_absent_is_none_rather_than_empty() -> None:
    """None says the arm was not run; an empty dict would say it ran and ranked nothing."""
    assert mean_ranks(LOG, "per_edge") is None


def test_a_section_with_no_rank_block_is_none() -> None:
    assert mean_ranks("=== sum ===\nno block here\n", "sum") is None


def test_a_trailing_mode_still_reads_to_the_end_of_the_file() -> None:
    """The last section has no following heading to stop at, which is its own boundary case."""
    assert mean_ranks(LOG, "discounted") == {"alpha": 7.7, "beta": 8.8}


@pytest.mark.skipif(not list(RESULTS.glob("*_statistics.log")),
                    reason="no statistics log to read")
def test_every_real_log_yields_distinct_numbers_per_mode() -> None:
    checked = 0
    for path in sorted(RESULTS.glob("*_statistics.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        got = {m: mean_ranks(text, m) for m in ("sum", "discounted", "per_edge")}
        present = {m: r for m, r in got.items() if r}
        if len(present) < 2:
            continue
        checked += 1
        values = list(present.values())
        assert values[0] != values[1], (
            f"{path.name}: two modes report identical mean ranks, which means one section "
            f"was read twice rather than two sections read once")
    assert checked, "no log holds two modes; this check inspected nothing"
    print(f"\n[ranks] {checked} log(s) with two or more modes, all distinct")
