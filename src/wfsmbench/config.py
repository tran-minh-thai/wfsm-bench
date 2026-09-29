"""Central parameter block — the default coefficients fixed in decision D3.

Every tunable constant lives here so a run is fully described by one object (and,
optionally, one YAML file). Grouped by the phase that consumes it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

# The fitness normalization anchor this study fixes, in pattern edges. Every experiment
# script uses this one value, so a fitness printed by one script means the same thing as a
# fitness printed by another and the tables can be read side by side. It is at least the
# largest maximum pattern size any stage runs (6 edges); it was set to 7 for a preset that
# has since been removed, and is kept at 7 because every recorded fitness uses it.
#
# Raising it re-scales every reported fitness, so it is a decision about the study as a
# whole, not a knob to turn per run. Raise it only to admit a larger pattern size, and then
# regenerate every result: numbers from two different anchors must never appear in one table.
# Params.size_reference keeps the smaller D3 default (6) so the unit tests stay hand-checkable
# against the documented default configuration.
STUDY_SIZE_REFERENCE = 7


@dataclass
class Params:
    # Fitness F(g) = alpha*Norm(WSUP) + beta*Norm(ATW) - gamma*Pen(g).
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.1
    # Penalty Pen(g) = eta1*|V| + eta2*|E|.
    eta1: float = 0.1
    eta2: float = 0.1
    # Weight aggregation for ATW (see measure.evaluation / measure.fitness).
    #   "sum"      - W(e) is the sum of matched edge weights (the original definition).
    #                ATW then grows with pattern size, which makes F rise almost
    #                monotonically with size: bigger is nearly always better.
    #   "per_edge" - W(e) is divided by the pattern's edge count, so ATW is the average
    #                weight PER matched edge and no longer scales with size. Weight quality
    #                and support (which falls with size) then trade off, giving an interior
    #                optimum instead of "as large as the bound allows".
    #   "discounted" - W(e) sorts the matched weights heaviest-first and discounts each by a
    #                further factor rho, so W(e) <= W_max/(1-rho) whatever the pattern's size.
    #                This is the mode the analysis in scripts/run_degeneracy.py asks for.
    #
    # "sum" and "per_edge" are the two ends of one family, and both ends are degenerate:
    # under "sum" the normalized ATW grows linearly with size, so the growth term outruns the
    # size penalty and the optimum sits at the size bound; under "per_edge" it is flat, so
    # falling support drives the optimum down to a single edge -- measured at a single-edge
    # optimum in all 21 configurations of the sweep. Neither is fixed by the penalty
    # coefficients, which only tilt a line and cannot reshape the growth term.
    #
    # What is needed is an ATW that saturates, and a discounted sum is bounded independently
    # of size, so it does. Measured on Cuneiform at a 3.3% support threshold -- the practical
    # setting every earlier experiment used -- the optimum moves inside the size range, with
    # the best fitness *positive* (0.317) rather than the negative value a heavy size penalty
    # buys. rho = 1 discounts nothing and reproduces "sum" exactly.
    weight_mode: str = "sum"
    # The discount factor, used only by weight_mode="discounted", and calibrated per dataset:
    # where the optimum leaves the size bound depends on how fast support falls with pattern
    # size, which is a property of the data. run_degeneracy.py locates it.
    #
    # 0.5 for Cuneiform, and the test that fixes it is stability under widening the bound
    # rather than the mere presence of an interior peak. At rho = 0.6 the optimum is interior
    # at a bound of 4 edges and back at the bound at 5, because the frontier dips at 4 and
    # recovers at 5 through a cycle-closing edge, which costs gamma*eta2 rather than
    # gamma*(eta1+eta2) -- the two-cost effect run_degeneracy derives. At rho <= 0.5 the
    # optimum is the same pattern with the same fitness (0.3169) at both bounds, which is what
    # an interior optimum has to mean: a property of the objective, not of where it was cut off.
    rho: float = 0.5
    # Rounds the objective the search navigates onto a grid of this width, leaving the
    # true objective untouched. It exists to make plateau structure an independent
    # variable: every dataset found so far ties plateaus to graph structure, so the
    # only way to ask which of the two moves a method ranking is to vary one on a
    # fixed dataset. Zero disables it.
    fitness_quantum: float = 0.0
    # Upper anchor for the ATW / WSUP normalizers, counted in pattern edges (decision D2).
    #
    # This must NOT be max_edges, which is what it used to be. max_edges bounds the SEARCH
    # SPACE; using it as the normalizer made F(g) a function of the search configuration, so
    # the same pattern scored differently under different bounds and the objective changed
    # silently with every size setting. The symptom was in the results: the exhaustive
    # optimum on BZR_MD was 0.6560 at max_size 3 and 0.6478 at max_size 4 -- impossible when
    # the feasible sets are nested, since every size-3 pattern is still feasible at size 4.
    # The optimum had not fallen; the denominator had grown. No table comparing fitness
    # across max_size means anything until this anchor is held fixed.
    #
    # Choose it once per study -- the largest max_edges the study uses -- and leave it there.
    # A value below max_edges is rejected: the heaviest patterns would clip to Norm = 1 and
    # become indistinguishable, so gbest would be decided by the penalty term alone.
    size_reference: int = 6
    # Pivot score S(v) = l1*deg + l2*wdeg + l3*cent + l4*lab.
    lambda1: float = 0.25
    lambda2: float = 0.25
    lambda3: float = 0.25
    lambda4: float = 0.25
    # PSO velocity update.
    omega: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    # Refuse to score a pattern that provably cannot be frequent, using the anti-monotonicity
    # of support: every subgraph of a frequent pattern is frequent, so a pattern with an
    # infrequent subgraph is infrequent and can never be reported. The exhaustive baseline
    # has always used this -- it is what makes enumeration finite -- and none of the searches
    # did, which is why they spend most of a run on patterns that cannot be an answer:
    # measured at a binding threshold (Cuneiform n=60, size 6, min_support 30, budget 180),
    # 44% of CLPSO's evaluations, 60% of the basic swarm's and 76% of the colony's went to
    # patterns below the threshold.
    #
    # Off by default. Turning it on changes what every method does, so it belongs in an
    # experiment as a condition to be measured against the unpruned run, not as a silent
    # improvement that makes the old numbers incomparable.
    prune_infrequent: bool = False
    # Search budget.
    swarm_size: int = 5
    iterations: int = 3
    top_k: int = 10
    min_support: int = 2
    # Top-K admission: 0.0 keeps the plain best-fitness set, higher values trade fitness
    # for structural novelty between the reported patterns (see measure.diversity).
    diversity_weight: float = 0.0
    # Dynamic multi-swarm topology (see pso.multiswarm). 1 sub-swarm == the basic star
    # topology; regrouping every few iterations lets information cross sub-swarms.
    num_subswarms: int = 4
    regroup_period: int = 2
    # What a search does when it stops finding anything new. Without a restart it simply
    # stops, and measurements showed most methods then refuse a larger budget entirely: on
    # Cuneiform the basic swarm scored 143 distinct patterns whether it was given 180
    # evaluations or 5400, the multi-swarm 181, the colony 172, the genetic search 79 -- of a
    # space holding about 1288. Restarting part of the population from fresh ground lets the
    # remaining budget be used; only when several restarts in a row find nothing does a run end.
    restart_fraction: float = 0.5
    max_barren_restarts: int = 3
    # CLPSO holds a particle's exemplar fixed until its personal best has failed to improve
    # this many rounds (Liang et al., 2006). The value only means anything relative to the
    # number of rounds a run gets, which is budget / swarm_size. The published 7 assumes runs
    # of thousands of rounds; on the ~9 rounds a 180-evaluation budget allows it pins the
    # exemplar for almost the whole run, and CLPSO degrades monotonically as it grows
    # (2.9% at 1, 10.1% at 7). Given enough rounds the published value is the better one --
    # at 1800 evaluations CLPSO reaches the optimum on every seed with a gap of 7. Default to
    # the short-run setting and raise it for long runs.
    clpso_refresh_gap: int = 1
    # Genetic baseline: share of the population carried over unchanged each generation.
    # This started at half the population, which left so little room for new individuals
    # that the search converged after about 50 distinct patterns -- a third of its budget --
    # and finished 11% short of the optimum. Carrying a single elite instead, the classic
    # setting, it reaches the optimum. A baseline is only informative at its own best
    # setting, so the default is the one that serves it best.
    elite_fraction: float = 0.05
    # Ant colony (see baselines.aco): trail vs heuristic exponents, evaporation, colony size.
    aco_alpha: float = 1.0
    aco_beta: float = 2.0
    aco_evaporation: float = 0.1
    num_ants: int = 10
    # How a particle's position is compared (see pso.structural).
    #   "template"   - Delta over edge-template multisets (decision D1 as written).
    #   "structural" - Delta over motif contexts (template plus endpoint degrees), which
    #                  distinguishes patterns the template multiset conflates.
    velocity_mode: str = "template"
    # Structural bounds.
    max_nodes: int = 6
    max_edges: int = 6
    max_velocity: int = 3
    # Pivot selection strategy (for the ablation study): one of
    # "scientific" (full S(v)), "degree", "random", "none".
    pivot_strategy: str = "scientific"

    _PIVOT_STRATEGIES = ("scientific", "degree", "random", "none")
    _WEIGHT_MODES = ("sum", "per_edge", "discounted")
    _VELOCITY_MODES = ("template", "structural")

    def __post_init__(self) -> None:
        if self.pivot_strategy not in self._PIVOT_STRATEGIES:
            raise ValueError(
                f"pivot_strategy must be one of {self._PIVOT_STRATEGIES}, "
                f"got {self.pivot_strategy!r}")
        if self.weight_mode not in self._WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {self._WEIGHT_MODES}, got {self.weight_mode!r}")
        if self.velocity_mode not in self._VELOCITY_MODES:
            raise ValueError(
                f"velocity_mode must be one of {self._VELOCITY_MODES}, "
                f"got {self.velocity_mode!r}")
        if not 0.0 < self.restart_fraction <= 1.0:
            raise ValueError(
                f"restart_fraction must be in (0, 1], got {self.restart_fraction!r}")
        if self.max_barren_restarts < 1:
            raise ValueError(
                f"max_barren_restarts must be >= 1, got {self.max_barren_restarts!r}")
        if self.clpso_refresh_gap < 1:
            raise ValueError(
                f"clpso_refresh_gap must be >= 1, got {self.clpso_refresh_gap!r}")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError(
                f"elite_fraction must be in (0, 1), got {self.elite_fraction!r}")
        if not 0.0 < self.aco_evaporation <= 1.0:
            raise ValueError(
                f"aco_evaporation must be in (0, 1], got {self.aco_evaporation!r}")
        if self.num_ants < 1:
            raise ValueError(f"num_ants must be >= 1, got {self.num_ants!r}")
        if self.num_subswarms < 1:
            raise ValueError(f"num_subswarms must be >= 1, got {self.num_subswarms!r}")
        if self.regroup_period < 1:
            raise ValueError(f"regroup_period must be >= 1, got {self.regroup_period!r}")
        if not 0.0 <= self.diversity_weight <= 1.0:
            raise ValueError(
                f"diversity_weight must be in [0, 1], got {self.diversity_weight!r}")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {self.rho!r}")
        if self.size_reference < 1:
            raise ValueError(f"size_reference must be >= 1, got {self.size_reference!r}")
        if self.size_reference < self.max_edges:
            raise ValueError(
                "size_reference is the fitness normalization anchor and has to cover the "
                f"search bound: got size_reference={self.size_reference} < "
                f"max_edges={self.max_edges}. Set it once, to the largest max_edges the "
                "study uses, so fitness stays comparable across sizes.")

    @staticmethod
    def from_dict(values: dict) -> "Params":
        known = {f.name for f in fields(Params)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        return Params(**values)

    @staticmethod
    def from_yaml(path: str | Path) -> "Params":
        return Params.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict:
        return asdict(self)
