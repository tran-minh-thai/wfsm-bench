"""The particle-swarm search loop (decisions D1-D4 in docs/DesignDecisions.md).

Reformulates weighted frequent subgraph mining as ``max F(g)`` over connected subgraphs
and solves it with a pivot-guided, discrete particle swarm. Each particle is a candidate
subgraph; its "velocity" is a sequence of graph-edit moves; it is pulled toward its own
best and the global best position.

Discrete PSO operators (D1):
- ``scale(moves, coeff, rng)`` — coeff (.) Delta: keep each move with probability
  ``min(1, coeff)``;
- ``compose(...)`` — Omega: concatenate inertia -> cognitive -> social, cancel opposing
  Grow/Shrink of the same template, keep one Repivot, cap the length at ``max_velocity``;
- ``velocity_update`` / ``apply_velocity`` — the velocity and position updates.

Randomness flows from a single seeded generator (D4), so a run is reproducible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import Params
from ..core.graph import Graph
from ..core.particle import Particle
from ..core.signature import canonical_signature
from ..match import match_database
from ..measure import Evaluation, discounted_weight, embedding_weight, fitness
from ..measure.diversity import select_diverse
from ..pivot import select_pivot
from ..preprocess.statistics import DatasetStats
from ..preprocess.template import EdgeTemplate
from . import structural
from .edits import (Grow, Move, Repivot, Shrink, apply_move, apply_velocity,
                    connected, diff, is_valid)


def _neighbors(particle: Particle, frequent: set[EdgeTemplate], params: Params) -> list[Particle]:
    """Valid single-edit neighbours: grow by each frequent template, or shrink an edge.
    Used to keep a particle exploring when its velocity would leave it in place."""
    result: list[Particle] = []
    for template in frequent:
        grown = apply_move(particle, Grow(template))
        if grown is not None and is_valid(grown, frequent, params):
            result.append(grown)
    pattern = particle.pattern
    own_templates = {EdgeTemplate.of(pattern.label(u), pattern.label(v), edge_label)
                     for (u, v), edge_label in pattern.edge_items()}
    for template in own_templates:
        shrunk = apply_move(particle, Shrink(template))
        if shrunk is not None and is_valid(shrunk, frequent, params):
            result.append(shrunk)
    return result


@dataclass(frozen=True)
class Candidate:
    particle: Particle
    evaluation: Evaluation
    fitness: float

    @property
    def signature(self) -> str:
        return canonical_signature(self.particle.pattern)


@dataclass
class Result:
    top_k: list[Candidate]
    gbest: Candidate | None
    # Every frequent candidate the search admitted, before the Top-K rule was applied.
    # Kept so a study can re-report the same run under a different admission rule without
    # searching again (the search itself does not depend on that rule).
    pool: list[Candidate] = field(default_factory=list)


def single_edge_deletions(pattern: Graph) -> list[Graph]:
    """Every connected pattern one edge smaller, with any node it isolates dropped.

    Disconnected results are skipped: the searches only ever score connected patterns, so
    nothing disconnected can be in the set this is checked against.
    """
    smaller: list[Graph] = []
    for edge in list(pattern.edges):
        edges = dict(pattern.edges)
        del edges[edge]
        node_labels = dict(pattern.node_labels)
        trimmed = Graph(node_labels, edges, pattern.graph_id)
        for isolated in [n for n in list(node_labels) if trimmed.degree(n) == 0]:
            del node_labels[isolated]
        candidate = Graph(node_labels, edges, pattern.graph_id)
        if candidate.node_labels and connected(candidate):
            smaller.append(candidate)
    return smaller


def infeasible(particle: Particle, stats: DatasetStats, params: Params) -> Candidate:
    """A pattern known to be infrequent, scored as though it matched nothing.

    Used only for patterns proven infrequent by anti-monotonicity, where the exact support
    is unknown but is certainly below the threshold. The fitness is the value an empty match
    would earn, which is a lower bound on the truth -- sound for a candidate that can never
    be reported, and it costs no database match to produce.
    """
    empty = Evaluation(support=0, atw=0.0, wsup=0.0)
    score = fitness(empty, particle.pattern.num_nodes(), particle.pattern.num_edges(),
                    stats, params)
    return Candidate(particle, empty, score)


def reported_best(pool: list[Candidate]) -> Candidate | None:
    """The best pattern a run reports, taken from the archive.

    A swarm's internal ``gbest`` is the attractor the particles are pulled toward, and it is
    deliberately unconstrained: a particle passing through an infrequent region should still
    be drawn back toward the fittest position seen, and rarer structures steer the search by
    being scored. But an infrequent pattern is not a mining result and must never be the
    answer a run hands back.

    Returning the attractor was doing exactly that. Measured on Cuneiform (n=60, size 5,
    min_support 30) the three swarms all reported a pattern of support 18 scoring 0.1540,
    against a true optimum of 0.1047 over the frequent patterns -- a "result" 47% better than
    the best that exists, because it was not a legal one. The baselines route their result
    through Archive.result(), which takes its best from the archive and was therefore always
    right, so the comparison was between one side that had to obey the support threshold and
    one side that did not. Every experiment reporting ``.gbest`` inherited that.

    The attractor is left alone -- only what a run reports changes, so search behaviour is
    exactly as measured before.
    """
    return max(pool, key=lambda c: c.fitness, default=None)


def select_top_k(candidates: list[Candidate], params: Params) -> list[Candidate]:
    """The reported Top-K. Every search method routes its result set through this, so a
    change of admission rule applies to all of them and the comparison stays fair."""
    scored = [(c.fitness, c.particle.pattern) for c in candidates]
    picked = select_diverse(scored, params.top_k, params.diversity_weight)
    return [candidates[i] for i in picked]


# --- discrete PSO operators ---------------------------------------------------

def scale(moves: list[Move], coeff: float, rng: np.random.Generator) -> list[Move]:
    """coeff (.) Delta: keep each move independently with probability min(1, coeff)."""
    probability = min(1.0, coeff)
    return [move for move in moves if rng.random() < probability]


def compose(inertia: list[Move], cognitive: list[Move], social: list[Move],
            max_velocity: int) -> list[Move]:
    """Omega: merge the three parts, cancel opposing template moves, cap the length.

    Three details decide whether the velocity actually moves the particle:

    - *Shrink before Grow.* A growth move is rejected outright once the pattern sits at its
      size bound, and on this objective particles reach that bound quickly. Emitting the
      removals first frees the budget so the additions can apply; the reverse order makes a
      velocity that is silently discarded in full.
    - *The pivot move keeps a slot.* It is placed at the front rather than appended past the
      cap, so truncation cannot discard the pivot mechanism, and it takes effect before the
      growth moves that expand from the pivot.
    - *Priority, not template id.* Templates are emitted in the order the parts contributed
      them (inertia, then cognitive, then social), so truncation keeps the highest-priority
      moves instead of the lexicographically smallest templates.
    """
    combined = [*inertia, *cognitive, *social]
    net: Counter = Counter()
    priority: dict[EdgeTemplate, int] = {}
    for position, move in enumerate(combined):
        if isinstance(move, Grow):
            net[move.template] += 1
        elif isinstance(move, Shrink):
            net[move.template] -= 1
        else:
            continue
        priority.setdefault(move.template, position)

    shrinks: list[Move] = []
    grows: list[Move] = []
    for template in sorted(net, key=lambda t: priority[t]):
        count = net[template]
        if count > 0:
            grows.extend(Grow(template) for _ in range(count))
        elif count < 0:
            shrinks.extend(Shrink(template) for _ in range(-count))

    resolved: list[Move] = [*shrinks, *grows]
    repivots = [move for move in combined if isinstance(move, Repivot)]
    if repivots:
        resolved.insert(0, repivots[-1])   # the social (global-best) pull wins the pivot
    return resolved[:max_velocity]


def velocity_update(particle: Particle, velocity: list[Move],
                    pbest: Particle, gbest: Particle,
                    params: Params, rng: np.random.Generator) -> list[Move]:
    """The velocity update, in whichever position space ``params.velocity_mode`` selects.

    The two modes differ only in what counts as a difference between particles: bare edge
    templates, or templates anchored to their motif context. Everything downstream -- the
    scaling, the composition order, the validity filter -- is the same.
    """
    difference, merge = (structural.diff_structural, structural.compose_structural) \
        if params.velocity_mode == "structural" else (diff, compose)
    inertia = scale(velocity, params.omega, rng)
    cognitive = scale(difference(particle, pbest), params.c1 * rng.random(), rng)
    social = scale(difference(particle, gbest), params.c2 * rng.random(), rng)
    return merge(inertia, cognitive, social, params.max_velocity)


def move_particle(particle: Particle, velocity: list[Move],
                  frequent: set[EdgeTemplate], params: Params) -> Particle:
    """Phi, in the mode the velocity was built for."""
    if params.velocity_mode == "structural":
        return structural.apply_velocity_structural(particle, velocity, frequent, params)
    return apply_velocity(particle, velocity, frequent, params)


# --- evaluation and search ----------------------------------------------------

class Evaluator:
    """Scores patterns, caching by canonical signature and counting distinct scorings.

    Matching a pattern against the database is the expensive step, and revisiting a pattern
    the search has already scored costs nothing once it is cached. Budgets are therefore
    counted in *distinct* patterns scored: without the cache a method that wanders back over
    old ground pays twice for it, and one that caches gets to see far more of the space for
    the same nominal budget -- measured at 82 distinct patterns for the swarm against 147 for
    the colony on the same allowance, which is not a fair comparison of search strategies.
    """

    def __init__(self, hosts: list[Graph], stats: DatasetStats, params: Params,
                 budget: int | None = None) -> None:
        self._hosts = hosts
        self._stats = stats
        self._params = params
        self._cache: dict[str, Candidate] = {}
        self._infrequent: set[str] = set()
        self.budget = budget
        self.spent = 0
        self.pruned = 0        # patterns skipped as provably infrequent, costing no budget

    @property
    def exhausted(self) -> bool:
        return self.budget is not None and self.spent >= self.budget

    def _provably_infrequent(self, pattern: Graph) -> bool:
        """True when some subgraph one edge smaller is already known to be infrequent.

        Support is anti-monotone: every subgraph of a frequent pattern is frequent, so a
        pattern with an infrequent subgraph cannot itself be frequent. Only single-edge
        deletions are checked, not every subgraph -- these searches move one edit at a time,
        so an infrequent ancestor is usually one deletion away, and the check costs a handful
        of signatures against a database match it saves. It never prunes a frequent pattern:
        a hit means an infrequent subgraph genuinely exists.
        """
        if not self._params.prune_infrequent or not self._infrequent:
            return False
        return any(canonical_signature(smaller) in self._infrequent
                   for smaller in single_edge_deletions(pattern))

    def score(self, particle: Particle) -> Candidate:
        signature = canonical_signature(particle.pattern)
        cached = self._cache.get(signature)
        if cached is not None:
            # Same pattern, possibly a different pivot; the objective ignores the pivot.
            return Candidate(particle, cached.evaluation, cached.fitness)
        if self._provably_infrequent(particle.pattern):
            # Scored as an empty match rather than measured. The true support is somewhere
            # below the threshold and this is its lower bound, which is all the search can
            # use it for: the pattern is unreportable either way, and the bound points away
            # from a region that has nothing in it. No budget is charged, because nothing was
            # matched -- that saving is the point.
            self.pruned += 1
            candidate = infeasible(particle, self._stats, self._params)
            self._cache[signature] = candidate
            self._infrequent.add(signature)
            return candidate
        self.spent += 1
        candidate = evaluate(particle, self._hosts, self._stats, self._params)
        self._cache[signature] = candidate
        if candidate.evaluation.support < self._params.min_support:
            self._infrequent.add(signature)
        return candidate


# A stochastic search can spend a round revisiting known patterns and still find something
# new in the next one, so a run only stops after several consecutive barren rounds. Stopping
# on the first one cut methods off at very different points and undid the equal budget.
STALL_ROUNDS = 5


def restart_count(params: Params) -> int:
    """How many population members a stall replaces -- never the whole population, so the
    best position found so far is always carried through."""
    return max(1, min(params.swarm_size - 1,
                      int(params.swarm_size * params.restart_fraction)))


def fresh_particles(count: int, frequent_list, stats: DatasetStats, params: Params,
                    rng: np.random.Generator) -> list[Particle]:
    """Single-edge particles drawn from random frequent templates -- the ground a stalled
    search restarts from."""
    picks = rng.integers(0, len(frequent_list), size=count)
    return [_seed_particle(frequent_list[int(i)], stats, params, rng) for i in picks]


def make_evaluator(hosts: list[Graph], stats: DatasetStats, params: Params,
                   max_evaluations: int | None = None) -> Evaluator:
    """The shared budget for every search: distinct patterns scored, defaulting to the
    swarm-equivalent allowance so all methods are given the same amount of real work."""
    budget = max_evaluations if max_evaluations is not None \
        else params.swarm_size * (1 + params.iterations)
    return Evaluator(hosts, stats, params, budget)


def evaluate(particle: Particle, hosts: list[Graph],
             stats: DatasetStats, params: Params) -> Candidate:
    embeddings = match_database(particle.pattern, hosts)
    # Which W(e) the mode wants. "sum" and "per_edge" both start from the plain total --
    # per_edge divides by the edge count later, in the fitness -- while "discounted" needs
    # the individual weights, which is why the matcher hands them over unaggregated.
    aggregate = (discounted_weight(params.rho) if params.weight_mode == "discounted"
                 else embedding_weight)
    evaluation = Evaluation.from_embeddings(embeddings, aggregate)
    score = fitness(evaluation, particle.pattern.num_nodes(),
                    particle.pattern.num_edges(), stats, params)
    return Candidate(particle, evaluation, score)


def _seed_particle(template: EdgeTemplate, stats: DatasetStats, params: Params,
                   rng: np.random.Generator) -> Particle:
    pattern = Graph.of(
        {0: template.low_label, 1: template.high_label},
        [(0, 1, template.edge_label)],
    )
    return Particle(pattern, select_pivot(pattern, stats, params, rng))


@dataclass
class _State:
    current: Candidate
    velocity: list[Move]
    pbest: Candidate


def optimize(hosts: list[Graph], stats: DatasetStats, params: Params, seed: int,
             initial: list[Particle] | None = None,
             known: list[Candidate] | None = None,
             max_evaluations: int | None = None) -> Result:
    """The discrete swarm search loop. ``initial`` overrides the edge-template seeding with given starting
    positions and ``known`` pre-loads the archive — together these let a warm-start
    strategy hand the swarm a better starting point without changing the search itself."""
    rng = np.random.default_rng(seed)
    frequent_list = stats.frequent_edge_templates(params.min_support)
    frequent = set(frequent_list)
    archive: dict[str, Candidate] = {}
    if not frequent_list:
        return Result(top_k=[], gbest=None)

    budget = max_evaluations if max_evaluations is not None \
        else params.swarm_size * (1 + params.iterations)
    evaluator = Evaluator(hosts, stats, params, budget)

    def remember(candidate: Candidate) -> None:
        # The output set is weighted *frequent* patterns, so only archive candidates
        # that clear the support threshold. Rarer structures still guide the search
        # (their low fitness steers particles away), but are not reported.
        if candidate.evaluation.support < params.min_support:
            return
        existing = archive.get(candidate.signature)
        if existing is None or candidate.fitness > existing.fitness:
            archive[candidate.signature] = candidate

    for candidate in known or []:
        remember(candidate)

    # Draw the starting templates in a seed-dependent order. Taking them in sorted order
    # gives every run the same initial swarm, so runs differ only in what happens after the
    # first iteration -- measured as one distinct initial swarm across five seeds.
    if not initial:
        order = list(rng.permutation(len(frequent_list)))
        frequent_list = [frequent_list[i] for i in order]

    # Initialize the swarm from frequent edge templates (round-robin over that order),
    # or from the supplied warm-start positions.
    states: list[_State] = []
    gbest: Candidate | None = None
    for i in range(params.swarm_size):
        if initial:
            start = initial[i % len(initial)]
        else:
            start = _seed_particle(frequent_list[i % len(frequent_list)], stats, params, rng)
        candidate = evaluator.score(start)
        remember(candidate)
        states.append(_State(current=candidate, velocity=[], pbest=candidate))
        if gbest is None or candidate.fitness > gbest.fitness:
            gbest = candidate

    # Iterate until the evaluation budget is spent rather than for a fixed count: with the
    # cache in place a revisited pattern is free, so a fixed iteration count would let one
    # method do far more real work than another for the same nominal allowance.
    # A sweep that scores nothing new means the swarm is only revisiting cached patterns,
    # so the budget can never be spent and the loop has to stop on its own.
    # A stalled swarm restarts its weakest members rather than stopping: measured, the swarm
    # otherwise scored the same 143 distinct patterns whether given 180 evaluations or 5400.
    # Only when several restarts in a row turn up nothing has the space really run out.
    stalled = 0
    barren_restarts = 0
    replace_count = restart_count(params)
    while not evaluator.exhausted and barren_restarts < params.max_barren_restarts:
        before = evaluator.spent
        for state in states:
            if evaluator.exhausted:
                break
            state.velocity = velocity_update(
                state.current.particle, state.velocity,
                state.pbest.particle, gbest.particle, params, rng)
            moved = move_particle(state.current.particle, state.velocity, frequent, params)
            # Anti-stagnation: if the velocity left the particle in place, jump to a random
            # neighbour so the swarm keeps exploring instead of freezing at its seed.
            if canonical_signature(moved.pattern) == state.current.signature:
                neighbours = _neighbors(state.current.particle, frequent, params)
                if neighbours:
                    moved = neighbours[int(rng.integers(len(neighbours)))]
            # The pivot stays as the velocity left it. Re-scoring it from the pattern here
            # would make the pivot a pure function of the pattern, so a Repivot move -- and
            # the social pull toward another particle's pivot -- could never persist.
            if not is_valid(moved, frequent, params):
                continue
            candidate = evaluator.score(moved)
            remember(candidate)
            # Standard PSO: the particle moves unconditionally; the personal and global
            # bests (and the archive) retain the best patterns seen. Moving through weaker
            # states is what lets a particle escape a local optimum.
            state.current = candidate
            if state.current.fitness > state.pbest.fitness:
                state.pbest = state.current
            if state.pbest.fitness > gbest.fitness:
                gbest = state.pbest
        if evaluator.spent > before:
            stalled = 0
            barren_restarts = 0
            continue
        stalled += 1
        if stalled < STALL_ROUNDS:
            continue
        # Keep the best positions, reseed the rest from fresh templates.
        stalled = 0
        barren_restarts += 1
        states.sort(key=lambda st: st.current.fitness, reverse=True)
        for state, particle in zip(
                states[len(states) - replace_count:],
                fresh_particles(replace_count, frequent_list, stats, params, rng)):
            candidate = evaluator.score(particle)
            remember(candidate)
            state.current = candidate
            state.velocity = []
            state.pbest = candidate

    pool = list(archive.values())
    # The attractor `gbest` may be an infrequent pattern; what a run reports may not.
    return Result(top_k=select_top_k(pool, params), gbest=reported_best(pool),
                  pool=pool)
