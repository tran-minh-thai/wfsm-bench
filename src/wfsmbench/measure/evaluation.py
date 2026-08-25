"""Weight-based evaluation of a pattern from its embeddings.

Follows this project's definitions, which differ from other WFSM
measures: the embedding weight is the *sum* of matched edge weights, and ATW is the
*mean of that sum over all embeddings across the database* (an absolute value, not a
normalized ratio).

    W(e)     = sum of the matched host edge weights of one embedding e
    ATW(g)   = mean of W(e) over every embedding of g in the database
    supp(g)  = number of distinct graphs that contain an embedding
    WSUP(g)  = supp(g) * ATW(g)

Input shape: ``embeddings_by_graph`` maps a host graph id to the list of embedding
weights W(e) found in that graph (only graphs with at least one embedding appear).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


def embedding_weight(matched_weights: Iterable[float]) -> float:
    """W(e): the total weight of one embedding."""
    return sum(matched_weights)


def discounted_weight(rho: float):
    """W_rho(e): the matched weights sorted heaviest-first and geometrically discounted.

    Each further edge counts ``rho`` times the one before it, so the total is bounded by
    ``W_max / (1 - rho)`` however many edges the pattern has. That bound is the point: it is
    what makes the normalized ATW saturate in pattern size instead of growing with it, which
    is the condition an interior optimum needs (see config.weight_mode and
    scripts/run_degeneracy.py). rho = 1 discounts nothing and gives back the plain sum.
    """
    def aggregate(matched_weights: Iterable[float]) -> float:
        return sum(w * rho ** i
                   for i, w in enumerate(sorted(matched_weights, reverse=True)))
    return aggregate


@dataclass(frozen=True)
class Evaluation:
    support: int
    atw: float
    wsup: float

    @staticmethod
    def from_embeddings(embeddings_by_graph: Mapping[int, list[list[float]]],
                        aggregate=embedding_weight) -> "Evaluation":
        """``embeddings_by_graph`` maps a host graph id to one weight list per embedding.

        The per-edge weights are kept rather than pre-summed because the aggregation is a
        scoring decision, not a matching one: which W(e) is wanted depends on the weight mode,
        and the matcher should not need to know.
        """
        per_embedding = [aggregate(weights)
                         for embeddings in embeddings_by_graph.values()
                         for weights in embeddings]
        support = sum(1 for embeddings in embeddings_by_graph.values() if embeddings)
        atw = sum(per_embedding) / len(per_embedding) if per_embedding else 0.0
        return Evaluation(support=support, atw=atw, wsup=support * atw)
