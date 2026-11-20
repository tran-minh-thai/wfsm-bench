# Design decisions

Fixed 2026-07-19, before the code was written. Where a decision has since been corrected the
correction is recorded beside it rather than replacing it, because the corrected version is
usually stronger than the one it replaces and the trail is what makes the work auditable.

Every constant here is a **default** and is configurable. The values are chosen so the terms
of the objective share a scale.

Decisions D1 to D4, D7 and D8 are cited from docstrings in `src/`. A decision about how the
work positions itself academically used to live here as D5; it belongs with the paper and was
moved there on 2026-08-26, which is why the numbering skips it.

## D1 — The velocity operator and scalar multiplication on edit sequences

**A velocity is an ordered list of primitive graph edits**: add an edge, expand from the
pivot, remove an edge, change the pivot, refine.

**The difference of two particles** is the edit sequence taking one to the other, read off
the symmetric difference of their edge sets plus a pivot change: an edge in the target but
not the source becomes an add, an edge in the source but not the target becomes a remove, and
a differing pivot becomes a pivot change.

**Scalar multiplication** discretises the classical PSO product: keep each edit of the
difference independently with probability `min(1, coeff)`, where the coefficient is the
inertia weight, or `c1·r1` for the cognitive pull, or `c2·r2` for the social pull, with `r1`
and `r2` drawn afresh per particle per iteration.

**The velocity operator** concatenates the three in priority order -- inertia, then
cognitive, then social -- cancels contradictions (an add and a remove of the same edge), and
truncates to a maximum length. Applying a velocity walks the sequence in order and **skips**
any edit that would violate connectivity, labelling, template validity or the size bounds at
the state reached so far, so the result is always a valid candidate.

*Why:* this is the standard discretisation of PSO for a combinatorial space -- keep or drop
each operation with a probability. The priority order and the length cap keep the search
stable instead of letting velocities grow without bound.

## D2 — Normalisation inside the objective

Min-max normalisation against a **fixed reference upper bound**, computed once during
preprocessing, rather than min-max against the current population:

- `W_max` is the largest edge weight in the (subset of the) collection;
- `E_ref` is a **constant of the study** (`Params.size_reference`, default 6) counted in
  pattern edges -- deliberately **not** `max_edges`;
- `ATW_ref = E_ref · W_max`, and `WSUP_ref = N_ref · ATW_ref` with `N_ref` the number of
  graphs considered;
- each normalised term is the raw term over its reference, clipped to `[0, 1]`.

*Why:* the objective must be **stationary** -- a given pattern always scores the same -- so
that comparing a particle's best against the global best across iterations is consistent and
reproducible. Normalising against the population would let the objective drift over time and
would break the meaning of personal and global best.

### D2-a — Correction of 2026-08-07: separate the anchor from the search bound

**The first version of D2 set `E_ref = max_edges`, and that was wrong.** `max_edges` bounds
the **search space**; using it as the denominator makes the objective a function of the
**search configuration** rather than of the pattern. The same three-node, two-edge pattern
scored:

| `max_edges` | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| objective, old anchor | 0.650 | 0.417 | 0.300 | 0.230 | 0.183 |
| objective, fixed anchor | 0.183 | 0.183 | 0.183 | 0.183 | 0.183 |

The symptom was already visible in the results: on one collection the exhaustive optimum was
**0.6560 at size bound 3** and **0.6478 at size bound 4**, which is impossible when the
feasible sets nest -- every size-3 pattern is still legal at size 4. The optimum had not
fallen; the denominator had risen.

**Decision:** `E_ref = Params.size_reference`, chosen **once for the whole study** and set to
the largest size bound the study uses. `Params.__post_init__` **refuses** a
`size_reference` below `max_edges`: below it the heaviest patterns clip to 1, tie with each
other, and the global best is decided by the penalty term alone.

*Consequence:* any table comparing objective values across size bounds is meaningful only
after re-running with the fixed anchor. Held by
`tests/test_measure.py::test_fitness_does_not_depend_on_the_search_bound`.

## D3 — Default coefficients

| group | coefficients | default | note |
|---|---|---|---|
| swarm | inertia, cognitive, social | 0.7, 1.5, 1.5 | standard PSO |
| objective | alpha, beta, gamma | 1.0, 0.5, 0.1 | weighted support leads, mean weight supports it, penalty is light |
| penalty | eta1, eta2 | 0.1, 0.1 | the penalty grows with size without overtaking the reward |
| pivot | lambda1..4 | 0.25 each | four criteria weighted equally |
| search | swarm, iterations, top-k | 5, 3, 10 | the light configuration of the original proposal |
| bounds | max nodes, max edges, velocity length | 6, 6, 3 | pattern size and edits per iteration |
| support | minimum support | low, e.g. 2 | loose enough to explore |

*Why:* every normalised term lies in `[0, 1]`, and the penalty coefficients are set so the
penalty stays a small share of the reward rather than dominating it.

**Note added 2026-08-26.** The search row records the proposal's light configuration. The
experiments reported in the paper sweep budgets far above it, and the budget grid is set per
run rather than from this default.

## D4 — Randomness and reproducibility

All randomness goes through a seeded generator; the seed is passed into each run and
recorded with it. Experiments run over many seeds. No global random state is used anywhere.

## D6 — Edge-label semantics on the complete-graph collections

The `_MD` collections are **complete** graphs: every pair of nodes carries an edge, the edge
label is a distance bin, and the edge weight is the distance itself. Decision: load **both**
the edge label and the weight from the source files, so a "subgraph" here means a
label-constrained substructure of a complete graph. Matching is expensive on such graphs, and
the cost is kept down by limiting the number of graphs, keeping patterns small, and pruning by
signature before matching. The label semantics must be stated wherever results on these
collections are reported.

*Why:* this is a loader and data-interpretation requirement; it changes no algorithm.

## D7 — Edge weights from the source collections

One reproducible rule: **the weight is the raw edge distance, min-max scaled per collection
to `[0.5, 5.0]`**, where the distance is the edge attribute if it is one-dimensional and its
Euclidean norm if it has several dimensions.

- On the one-dimensional collection this **reproduces exactly** the table of the authors'
  earlier, unpublished report (mean 1.519271, min 0.5, max 5.0). Corrected 2026-11-20: this note
  called that table "published"; the report was never published.
- On the two-dimensional collection roughly 71% of attribute vectors are zero, and **no
  simple transform reproduces** that report's mean of 2.418317 -- the report does not state
  how it preprocessed those weights. The general rule is used instead. Structure (graph, node
  and edge counts) still matches that table; only the mean weight differs, and that
  difference is reported wherever the collection is used.

**Note added 2026-08-26.** This scaling is why every collection shares the same weight
interval, which is a fact about the pipeline rather than about any collection. Min-max
scaling is a strictly increasing map, so the non-dominated set is unaffected by it while a
scalarised answer is not.

## D8 — Pattern edge weights for the pivot score

The weighted-degree term of the pivot score needs edge weights **on the pattern**, but an
abstract pattern carries none: the weights live on the host graphs and are used during
matching. Decision: each pattern edge takes a **representative weight, the mean weight of
that edge template across the collection**.

*Why:* it reflects the typical interaction strength around a node for that kind of edge, it
is **stationary** -- depending only on collection statistics and pattern structure -- and it
needs no matching to compute. Ties when selecting a pivot go to the lower node id.

## D9 — Diversity-constrained top-k reporting

The archive originally took the top k **by objective value alone**. Because high-scoring
patterns tend to be near-duplicate variants of one structure, the reported set repeated
itself. A `diversity_weight` parameter was added, **defaulting to 0.0**, which preserves the
old behaviour exactly. Above zero, the top k is selected by maximal marginal relevance: each
round takes the candidate maximising

    (1 − weight)·normalised objective + weight·(distance to the nearest already-chosen pattern)

with distance the Jaccard distance on edge-template sets and the objective min-max normalised
over the candidate pool, so both terms share the scale `[0, 1]`. The first round always takes
the best pattern.

*Why:* it separates **search quality** from **the reporting rule**. At zero every earlier
result is reproducible; above zero it is an independent ablation axis. The selection lives in
one function that **every method goes through**, swarm and baselines alike, so comparisons
stay like for like.

## D10 — Seeding the swarm from a shallow beam

Two methods fail in opposite ways: a deterministic beam reaches a high objective value but
returns near-duplicates, while the swarm explores widely but spends much of its budget
climbing from single-edge seeds. The hybrid runs a **shallow** beam and hands its frontier to
the swarm as starting positions.

**Budget fairness is mandatory.** The evaluations the beam consumes are converted into
equivalent iterations and **subtracted** from the swarm's, so the hybrid and the plain swarm
receive the same total evaluation budget. The optimiser takes optional starting positions and
a pre-loaded archive; **the search itself is unchanged**, so any observed difference comes
from the starting point alone.
