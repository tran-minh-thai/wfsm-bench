"""Mean ranks for one cell, read from the statistics log that produced them.

From the log rather than recomputed: recomputing would be a second implementation of a
result already established by the test that was run, free to disagree with it. Two renderers
and one generator all quote these ranks, and each had grown its own copy of this parse --
including one that carried a docstring warning against exactly that.

Path resolution stays with the callers, since they name their logs differently. What is
shared is the part that can be wrong in the same way twice: finding the block for one
objective mode and reading the numbers out of it.
"""

from __future__ import annotations

import re

# A line of the log's mean-rank block: four spaces, a method name, its mean rank.
ENTRY = re.compile(r"^\s{4}(\w+)\s+([\d.]+)$", re.M)

# The start of any objective mode's section. Used to END one mode's block, which is the part
# both earlier copies left implicit: they relied on splitting the whole file at every
# occurrence of the heading text and taking the segment between two of them. That happens to
# be right while the file holds exactly the modes it holds, and it silently reads the NEXT
# mode's numbers the moment anything about that changes -- which is what it did.
SECTION = re.compile(r"^=== \w+ ===$", re.M)


def mean_ranks(text: str, mode: str) -> dict[str, float] | None:
    """The mean rank per method for one objective mode, or None if that arm was not run.

    None and an empty dict mean different things and must not be conflated: the first says
    the arm is absent from this log, the second would say it ran and ranked nothing.
    """
    marker = f"=== {mode} ==="
    start = text.find(marker)
    if start < 0:
        return None
    # Bound the section explicitly at the next mode heading, so the entries below can only
    # be this mode's.
    after = text[start + len(marker):]
    following = SECTION.search(after)
    section = after[:following.start()] if following else after

    head, _, body = section.partition("mean ranks")
    if not _:
        return None
    return {name: float(value) for name, value in ENTRY.findall(body)}
