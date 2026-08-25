"""Fitness function F(g) and its parts (proposal eqs 18-19; decisions D2, D3).

    Pen(g)  = eta1*|V| + eta2*|E|
    Norm(z) = clip(z / z_ref, 0, 1)   with fixed reference anchors from DatasetStats (D2)
    F(g)    = alpha*Norm(WSUP) + beta*Norm(ATW) - gamma*Pen(g)

Fixed reference anchors keep the fitness stationary: a given pattern always scores the
same F, so personal-best / global-best comparisons stay consistent across iterations.

"Fixed" has to mean fixed against the search configuration as well as against the
population. The anchors are scaled by ``Params.size_reference``, a constant of the study,
never by ``max_edges``, which bounds the search space -- see the note in config.py. This is
what makes a fitness comparable across a sweep over maximum pattern size.
"""

from __future__ import annotations

from ..config import Params
from ..preprocess.statistics import DatasetStats
from .evaluation import Evaluation


def penalty(num_nodes: int, num_edges: int, params: Params) -> float:
    return params.eta1 * num_nodes + params.eta2 * num_edges


def norm(value: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return min(1.0, max(0.0, value / reference))


def fitness(evaluation: Evaluation,
            num_nodes: int,
            num_edges: int,
            stats: DatasetStats,
            params: Params,
            num_graphs_ref: int | None = None) -> float:
    atw, wsup, atw_ref, wsup_ref = _weight_terms(
        evaluation, num_edges, stats, params, num_graphs_ref)
    score = (
        params.alpha * norm(wsup, wsup_ref)
        + params.beta * norm(atw, atw_ref)
        - params.gamma * penalty(num_nodes, num_edges, params)
    )
    if params.fitness_quantum > 0.0:
        # Deliberately coarsens what the search can tell apart. Patterns falling in one
        # cell become indistinguishable to every method at once, so a comparison under a
        # quantum is still fair -- what changes is the landscape, not anyone's advantage.
        # The true value is recoverable by scoring again with the quantum at zero, which
        # is how a run's result is reported: the search is blind inside a plateau, the
        # measurement of what it returned is not.
        return round(score / params.fitness_quantum) * params.fitness_quantum
    return score


def _weight_terms(evaluation: Evaluation, num_edges: int, stats: DatasetStats,
                  params: Params, num_graphs_ref: int | None):
    """The ATW/WSUP values and their normalization anchors under the active weight mode.

    In "sum" mode these are the raw evaluation values against size-scaled anchors. In
    "per_edge" mode both the value and its anchor are divided by the edge count, so ATW
    becomes an average weight per edge (size-neutral) and WSUP = support * that average.
    """
    if params.weight_mode == "discounted":
        atw_ref = stats.atw_reference_discounted(params.size_reference, params.rho)
        n = stats.num_graphs if num_graphs_ref is None else num_graphs_ref
        return evaluation.atw, evaluation.wsup, atw_ref, n * atw_ref
    if params.weight_mode == "per_edge" and num_edges > 0:
        atw = evaluation.atw / num_edges
        wsup = evaluation.support * atw
        atw_ref = stats.max_edge_weight
        wsup_ref = (stats.num_graphs if num_graphs_ref is None else num_graphs_ref) * atw_ref
        return atw, wsup, atw_ref, wsup_ref
    return (evaluation.atw, evaluation.wsup,
            stats.atw_reference(params.size_reference),
            stats.wsup_reference(params.size_reference, num_graphs_ref))
