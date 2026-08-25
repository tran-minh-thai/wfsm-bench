"""Ant colony optimization for weighted frequent subgraph mining.

ACO belongs to a different family than the swarm searches: it is *constructive*. An ant does
not move an existing pattern around, it builds one edge at a time, choosing each extension
with a probability that mixes a learned pheromone trail with a static heuristic. That maps
onto pattern growth directly, without the position/velocity encoding a particle swarm needs
-- which is why it is worth comparing against here rather than as yet another swarm variant.

    P(t) proportional to  tau[t]^alpha * eta[t]^beta

``tau`` is the pheromone on an edge template, reinforced when it appears in a good pattern
and evaporating otherwise; ``eta`` is a fixed desirability, taken as the template's dataset
support times its mean weight, which is the information available before any search.

Every pattern an ant passes through is a valid candidate, so all of them are scored and
offered to the archive. Cost is counted the same way as for the other fixed-budget methods --
one evaluation per pattern scored -- and construction stops when the budget is spent, so the
comparison against the swarm and the beam stays like-for-like.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..core.signature import canonical_signature
from ..preprocess.statistics import DatasetStats
from ..preprocess.template import EdgeTemplate
from ..pso.optimizer import STALL_ROUNDS, Result, make_evaluator
from ._common import Archive, seed_particles, valid_children


def heuristic(stats: DatasetStats, templates: list[EdgeTemplate]) -> dict[EdgeTemplate, float]:
    """eta: how attractive a template looks before any search -- how many graphs contain it
    times how heavy it typically is. Scaled to (0, 1] so alpha/beta stay comparable."""
    raw = {t: (stats.edge_template_support.get(t, 0) / max(1, stats.num_graphs))
              * max(stats.template_weight(t), 1e-9)
           for t in templates}
    high = max(raw.values(), default=1.0) or 1.0
    return {t: max(value / high, 1e-6) for t, value in raw.items()}


def _added_template(parent: Graph, child: Graph) -> EdgeTemplate | None:
    """The template of the single edge that ``child`` adds to ``parent``."""
    parent_edges = set(parent.edges)
    for (u, v), edge_label in child.edge_items():
        if (u, v) not in parent_edges:
            return EdgeTemplate.of(child.label(u), child.label(v), edge_label)
    return None


def _choose(weights: list[float], rng: np.random.Generator) -> int:
    total = sum(weights)
    if total <= 0:
        return int(rng.integers(len(weights)))
    return int(rng.choice(len(weights), p=[w / total for w in weights]))


def mine_aco(hosts: list[Graph], stats: DatasetStats, params: Params, seed: int,
             max_evaluations: int | None = None) -> Result:
    """Ant colony search. Same signature and Result as the other baselines."""
    rng = np.random.default_rng(seed)
    frequent_list = stats.frequent_edge_templates(params.min_support)
    frequent = set(frequent_list)
    if not frequent_list:
        return Result(top_k=[], gbest=None)

    evaluator = make_evaluator(hosts, stats, params, max_evaluations)
    budget = evaluator.budget
    eta = heuristic(stats, frequent_list)
    tau: dict[EdgeTemplate, float] = {t: 1.0 for t in frequent_list}
    archive = Archive(params)

    def desirability(template: EdgeTemplate) -> float:
        return (tau.get(template, 1e-6) ** params.aco_alpha
                * eta.get(template, 1e-6) ** params.aco_beta)

    def score(particle: Particle) -> float:
        """Score one pattern through the shared evaluator and offer it to the archive."""
        candidate = evaluator.score(particle)
        archive.add(candidate)
        return candidate.fitness

    seeds = seed_particles(frequent_list)
    stalled = 0
    barren_restarts = 0
    while not evaluator.exhausted and barren_restarts < params.max_barren_restarts:
        # A colony round that scores nothing new means every reachable pattern has already
        # been paid for, so the budget can never be spent and the loop must stop here.
        before = evaluator.spent
        walks: list[tuple[float, list[EdgeTemplate]]] = []
        for _ in range(params.num_ants):
            if evaluator.exhausted:
                break
            weights = [desirability(t) for t in frequent_list]
            current = seeds[_choose(weights, rng)]
            used = [EdgeTemplate.of(current.pattern.label(u), current.pattern.label(v), label)
                    for (u, v), label in current.pattern.edge_items()]
            best = score(current)
            while not evaluator.exhausted:
                children = valid_children(current, frequent, params)
                if not children:
                    break
                added = [_added_template(current.pattern, child.pattern) for child in children]
                pick = _choose([desirability(t) if t else 1e-6 for t in added], rng)
                current = children[pick]
                if added[pick] is not None:
                    used.append(added[pick])
                best = max(best, score(current))
            walks.append((best, used))

        # Evaporate everywhere, then reinforce the templates of this colony's best walk.
        for template in tau:
            tau[template] *= (1.0 - params.aco_evaporation)
        if walks:
            best_score, best_used = max(walks, key=lambda w: w[0])
            for template in best_used:
                tau[template] = tau.get(template, 0.0) + max(best_score, 0.0)
        # A floor keeps every template reachable, which is what stops the colony from
        # collapsing onto one trail after a few reinforcements.
        for template in tau:
            tau[template] = max(tau[template], 1e-3)

        if evaluator.spent > before:
            stalled = 0
            barren_restarts = 0
            continue
        stalled += 1
        if stalled < STALL_ROUNDS:
            continue
        # The trail has converged onto ground already covered; forgetting it sends the ants
        # out on the heuristic alone, which is this method's form of a restart.
        stalled = 0
        barren_restarts += 1
        tau = {t: 1.0 for t in frequent_list}

    return archive.result()
