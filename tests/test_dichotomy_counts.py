"""The counts behind the growth-dichotomy paragraph, checked against hand-built sequences.

Each mode has a strict case: the sum mode rises when every step gains more than the dearest step
cost, gamma (eta1 + eta2), and adds at most one vertex; the per_edge mode falls when every step
gains less than the cheapest, gamma eta2, and loses no vertex. The real counts are printed as
well, with their denominators.
"""
from __future__ import annotations

from pathlib import Path

from wfsmbench.measure import verification as V

RESULTS = Path(__file__).resolve().parents[1] / "results"
HI = V.GAMMA * (V.ETA1 + V.ETA2)
LO = V.GAMMA * V.ETA2


def rows(points: list[tuple[int, int, float]]) -> list[dict]:
    return [{"edges": str(k), "nodes": str(v), "G": repr(g)} for k, v, g in points]


def test_steps_are_read_between_consecutive_best_patterns() -> None:
    series = rows([(1, 2, 0.10), (2, 3, 0.16), (3, 3, 0.20)])
    steps = V.dichotomy_steps(series, "sum")
    assert [dv for _dg, dv in steps] == [1, 0]
    assert abs(steps[0][0] - 0.06) < 1e-12 and abs(steps[1][0] - 0.04) < 1e-12
    pe = V.dichotomy_steps(series, "per_edge")
    assert abs(pe[0][0] - V.SIZE_REFERENCE * (0.16 / 2 - 0.10 / 1)) < 1e-12


def test_the_sum_case_needs_every_step_above_the_dearest_cost_and_at_most_one_vertex() -> None:
    assert V.strict_case_holds([(HI + 0.001, 1), (HI + 0.001, 0)], "sum")
    assert not V.strict_case_holds([(HI + 0.001, 1), (HI, 0)], "sum")          # equal is not above
    assert not V.strict_case_holds([(HI + 0.001, 1), (HI - 0.001, 0)], "sum")
    assert not V.strict_case_holds([(HI + 0.5, 2)], "sum")                     # two new vertices
    assert V.strict_case_holds([], "sum")                                      # no step: vacuous


def test_the_per_edge_case_needs_every_step_below_the_cheapest_cost_and_no_vertex_lost() -> None:
    assert V.strict_case_holds([(LO - 0.001, 1), (-0.2, 0)], "per_edge")
    assert not V.strict_case_holds([(LO, 0)], "per_edge")
    assert not V.strict_case_holds([(-0.2, -1)], "per_edge")


def test_the_real_counts_have_denominators_and_stay_inside_them(capsys) -> None:
    for mode in ("sum", "per_edge"):
        counts = V.dichotomy_counts(RESULTS, mode)
        assert counts["cells"] == V.dichotomy_cells(RESULTS, mode) >= 10, counts
        assert all(0 <= counts[k] <= counts["cells"] for k in ("strict", "predicted", "at_bound"))
        print(f"[dichotomy/{mode}] {counts}")
    assert V.dichotomy_counts(RESULTS, "per_edge")["at_bound"] == 0   # the per_edge optimum is one edge


def test_counts_on_a_built_results_tree(tmp_path) -> None:
    """Three cells at bound 4: optimum with K vertices but fewer edges, with K edges, and inside."""
    frontier = ["dataset,graph_limit,max_size,min_support,edges,nodes,G"]
    landscape = ["dataset,graph_limit,size_bound,support_fraction,min_support,weight_mode,"
                 "optimum_edges,optimum_nodes"]
    rising = [(1, 2, 0.10), (2, 3, 0.20), (3, 4, 0.30), (4, 4, 0.40)]     # every step above HI
    stalling = [(1, 2, 0.10), (2, 3, 0.20), (3, 4, 0.205), (4, 4, 0.21)]  # last steps below
    for support, points, edges, nodes in ((2, rising, 3, 4), (6, rising, 4, 4), (12, stalling, 2, 3)):
        frontier += [f"Cuneiform,60,4,{support},{k},{v},{g}" for k, v, g in points]
        landscape.append(f"Cuneiform,60,4,0.1,{support},sum,{edges},{nodes}")
    (tmp_path / "degeneracy_frontier_cuneiform.csv").write_text("\n".join(frontier) + "\n")
    (tmp_path / "weight_mode_landscape.csv").write_text("\n".join(landscape) + "\n")
    counts = V.dichotomy_counts(tmp_path, "sum")
    assert counts["cells"] == 3 and counts["strict"] == 2
    assert counts["at_bound"] == 1      # only K edges is at the bound; K vertices alone is not
