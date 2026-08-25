"""Multi-objective view of a result set: Pareto fronts, hypervolume, and coverage.

The fitness function collapses three competing goals into one number. That is what the
search optimizes, but it is not the only way to judge a result *set*: a population method
may return a wider spread of good trade-offs than a greedy trajectory even when its single
best pattern is no better. The multi-objective literature on subgraph mining measures
exactly that, with the hypervolume ratio and the coverage (C-) metric, so those are what we
compute here.

Objectives, all stated as maximization so one code path handles them:

    (WSUP, ATW, -(|V| + |E|))

WSUP and ATW pull toward larger, heavier patterns; the negated size pulls the other way,
which is the conflict that makes a front rather than a single point.

- ``pareto_front`` keeps the non-dominated candidates.
- ``hypervolume`` is the exact volume dominated by a front, measured against a reference
  (nadir) point, computed by slicing one objective at a time.
- ``hypervolume_ratio`` (HVR) divides a front's hypervolume by that of a reference front --
  the true front from exhaustive enumeration where it is tractable, otherwise the union of
  every method's front. HVR is 1.0 when a method recovers the whole reference front.
- ``coverage`` is the C-metric: the fraction of one front weakly dominated by another.

Objectives are min-max normalized against the reference front before any volume is taken,
so a hypervolume is comparable across datasets and lies in [0, 1].
"""

from __future__ import annotations

from ..core.graph import Graph
from .evaluation import Evaluation

Point = tuple[float, ...]


def objectives(evaluation: Evaluation, pattern: Graph) -> Point:
    """The maximization objective vector of one pattern."""
    return (evaluation.wsup,
            evaluation.atw,
            -float(pattern.num_nodes() + pattern.num_edges()))


def dominates(a: Point, b: Point) -> bool:
    """True if ``a`` is at least as good as ``b`` everywhere and strictly better somewhere."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(points: list[Point]) -> list[Point]:
    """The non-dominated subset, duplicates removed, in a deterministic order.

    Compares each candidate against the accepted front rather than against every other
    point, which is the difference between O(n^2) and O(n * |front|). On a pool of 42,523
    patterns with a front of a few dozen that is the difference between hours and seconds.

    Two facts make it exact rather than approximate, and both are needed:

    1. Scanning in descending lexicographic order, a later point can never dominate an
       earlier one. If p precedes q then p >= q lexicographically; for q to dominate p it
       would need q >= p in every objective, which together forces q == p, and duplicates
       are already gone.
    2. Dominance is transitive, so a point rejected because some earlier point dominated it
       cannot be the only thing standing between a later point and acceptance -- whatever
       dominated the rejected point dominates the later one too, and by induction something
       in the accepted front does.

    Output order is the same descending order the previous implementation produced, so this
    is a speed change and not a behaviour change; ``test_pareto_front_matches_naive`` holds
    it to that against a brute-force reference.
    """
    front: list[Point] = []
    for point in sorted(set(points), reverse=True):
        if not any(dominates(kept, point) for kept in front):
            front.append(point)
    return front


def normalize(points: list[Point], bounds: list[tuple[float, float]]) -> list[Point]:
    """Min-max each objective into [0, 1] using ``bounds``; a flat objective maps to 1.0."""
    scaled: list[Point] = []
    for point in points:
        row = []
        for value, (low, high) in zip(point, bounds):
            span = high - low
            row.append(1.0 if span <= 0 else min(1.0, max(0.0, (value - low) / span)))
        scaled.append(tuple(row))
    return scaled


def bounds_of(points: list[Point]) -> list[tuple[float, float]]:
    """Per-objective (min, max) over ``points`` -- the normalization anchors."""
    return [(min(p[i] for p in points), max(p[i] for p in points))
            for i in range(len(points[0]))]


def hypervolume(points: list[Point], reference: Point) -> float:
    """Exact volume dominated by ``points`` and bounded below by ``reference``.

    Slices the last objective: sort descending on it, and for each prefix of points add the
    (d-1)-dimensional volume of that prefix times the slab thickness down to the next
    distinct level. Exact for any dimension; the fronts here are small enough for it.
    """
    usable = [p for p in points if all(v > r for v, r in zip(p, reference))]
    if not usable:
        return 0.0
    if len(reference) == 1:
        return max(p[0] for p in usable) - reference[0]

    ordered = sorted(usable, key=lambda p: p[-1], reverse=True)
    total = 0.0
    for index, point in enumerate(ordered):
        lower = ordered[index + 1][-1] if index + 1 < len(ordered) else reference[-1]
        thickness = point[-1] - lower
        if thickness <= 0:
            continue
        prefix = [p[:-1] for p in ordered[:index + 1]]
        total += hypervolume(prefix, reference[:-1]) * thickness
    return total


def hypervolume_ratio(front: list[Point], reference_front: list[Point],
                      margin: float = 0.1) -> float:
    """HVR: the front's hypervolume as a fraction of the reference front's.

    Both fronts are normalized with the reference front's bounds, so a method that recovers
    the reference front scores 1.0 and one that recovers none of it scores 0.0.

    ``margin`` places the reference point that far *outside* the normalized nadir. Without
    it the reference front's own extreme points sit exactly on the nadir and contribute no
    volume, which would credit a single interior point with the whole hypervolume. Shifting
    the corner outward keeps every point of the front contributing.
    """
    if not front or not reference_front:
        return 0.0
    bounds = bounds_of(reference_front)
    corner = tuple(-margin for _ in bounds)
    reference_volume = hypervolume(normalize(reference_front, bounds), corner)
    if reference_volume <= 0:
        return 0.0
    return hypervolume(normalize(front, bounds), corner) / reference_volume


def coverage(front_a: list[Point], front_b: list[Point]) -> float:
    """C-metric C(A, B): the fraction of ``front_b`` weakly dominated by some point of A.

    C(A,B) = 1 with C(B,A) < 1 means A fully covers B. The measure is asymmetric, so both
    directions are worth reporting.
    """
    if not front_b:
        return 0.0
    covered = sum(1 for b in front_b
                  if any(all(x >= y for x, y in zip(a, b)) for a in front_a))
    return covered / len(front_b)
