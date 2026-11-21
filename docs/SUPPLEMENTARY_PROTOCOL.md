# Supplementary experiments: protocol fixed before the confirmatory runs

Written and committed on 2026-11-21, before any of the runs below. Anything changed after a
result has been seen is reported separately, under "Deviations", with the reason.

## Common rules

- **Cells.** The 16 selector configurations of `results/selector_regret_allpairs.csv`
  (dataset, weight mode, size bound K, support threshold), every budget pair (b_r, b_d) with
  b_r < b_d that the table scores (61 cells), and the same seeds: for each cell, the seeds
  present at both budgets for every method; race on the even ones, pay on the odd ones
  (n_r = n_d = n).
- **Objective.** Every reported fitness is the true objective F (no quantum), computed by
  `scripts/run_weight_mode.py::_job` with `params_for`, unchanged.
- **Gap.** gap = 100 (F* - F) / |F*|, where F* is the exhaustive optimum of the configuration
  (`optimum_fitness` of the landscape artefacts) and F the mean fitness over the paying seeds.
  It is on one scale for every budget. Regret keeps its current definition (against the best
  method at b_d on the paying seeds) and may be negative above b_d.
- **Machine.** Fitness, pattern and evaluation counts are deterministic given the commit, the
  data, the seed and the library versions. They run on the development machine and go to
  `results-invariant/revision-2026-11-21/`. Before any new run is trusted, the runner re-runs a
  sample of recorded runs and refuses if one fitness differs. No timing is reported from these
  runs.
- **Provenance.** Every artefact has a JSON file beside it with the command, the commit, the
  sha256 of the script, Python and library versions, and the data files' sha256.
- **Resume.** Runs are written one row at a time; a rerun of the same command skips the rows
  already present.

## T1. Fixed rules at the selector's cost

Cost of the selector in a cell: T_race = M n_r b_r + n_d b_d with M = 7.

- (a) One long run: each paying seed runs one search at b_eq = T_race / n_d = b_d + M b_r.
- (b) Restarts, best of k: each paying seed s runs k = floor(T_race / (n_d b_d)) = 1 + floor(M b_r / b_d)
  searches at b_d and keeps the best fitness. One of the k is the recorded run on seed s; the
  other k - 1 use seeds 1000 + 10 s + j, j = 1..k-1, which no race or paying run uses.
- All 7 methods, both variants. Reported per cell and per configuration: gap of the selector,
  of each fixed rule at b_d, at (a) and at (b).
- **Best fixed rule at equal cost**: the single (method, variant) with the lowest mean gap over
  the 61 cells, and separately over the 16 configurations. It is chosen in hindsight, which
  favours the fixed rules; it is reported as such.

## T2. Successive halving at the selector's cost

- Round 1 runs the 7 methods at b_0 on the racing seeds and keeps the ceil(m/2) best by mean
  fitness (7, 4, 2, 1); each later round doubles the budget, capped at b_d. Ties are broken by
  method name, in alphabetical order. The survivor is paid at b_d on the paying seeds (the
  recorded runs).
- b_0 is the largest integer with n (7 b_0 + 4 min(2 b_0, b_d) + 2 min(4 b_0, b_d)) <= M n b_r,
  so the total cost does not exceed T_race. The realized cost is reported per cell.
- Gap and regret as in T1, on the same cells.
- No racer with elimination (Hoeffding or F-Race style) exists in the repository; none is
  written for this. Reported as skipped.

## T3. Statistics per configuration

- Unit: the configuration (16). Each configuration's value is the mean over its budget pairs.
- Selector against each fixed rule at b_d, at (a), at (b), against the best fixed rule at
  equal cost (T1) and against successive halving (T2), on gap. Difference = gap of the rival
  minus gap of the selector (positive: the selector is closer to the optimum).
- Mean difference with a 95% percentile bootstrap interval: 10,000 resamples of the
  configurations with replacement, seed 20261121.
- Wilcoxon signed-rank test, two-sided, scipy default handling of zero differences (they are
  dropped and their number reported), Holm correction over all comparisons of the table.
- The same table without the Cuneiform configurations (9 left).

## T4. The growth dichotomy on the five molecular datasets

- Exhaustive landscape in `sum` and `per_edge` for BZR_bond, COX2_bond, DHFR_bond, QM9_bond and
  alchemy_full_bond, first 200 graphs, K = 4 and 5, support 2 (10 cells, the cells of the
  frontier artefacts `results/degeneracy_frontier_*.csv`).
- Prediction from the `sum` frontier with `verification.py` (S = 7), measured optimum from the
  new landscape. Per dataset: cells meeting the strict case, cells predicted correctly, cells
  whose measured optimum has an intermediate edge count (strictly between the pool's smallest
  and largest).
- Null "always a one-edge pattern": the number of cells it predicts correctly, over the same
  24 cells (14 Cuneiform, 10 molecular), in `per_edge`.

## T5. A real ball bound in the enumerator

- Front membership pruning in a breadth-first enumerator of frequent patterns, `sum` mode,
  objectives (WSUP, ATW, -(v + k)). A pattern g with k < K edges is not expanded when a member
  of the front built so far dominates the upper vector
  (supp(g) U, U, -(v + k)), U = max over embeddings phi of g of (A(phi) + rem(phi)).
- Remainders, with d = K - k:
  - global: rem = d w_max;
  - host graph: the sum of the d largest edge weights of the host graph of phi, edges of phi
    excluded;
  - ball: the sum of the d largest edge weights among the edges of the host graph with both
    endpoints within distance d of the vertices of phi, edges of phi excluded.
- Datasets of Table 10 (Cuneiform, QM9_bond, BZR_bond, alchemy_full_bond), first 200 graphs,
  K = 4, support 2. Per bound: patterns whose expansion is cut, evaluations, and over every
  non-leaf pattern the ratio of the host-graph U to the ball U (mean and 10th percentile).
- Soundness: the front with pruning must equal the front without pruning, for every bound; the
  run refuses otherwise.
- **Criterion 1, threshold fixed now: tau = 0.9.** Predicted "the ball bound is no tighter"
  when the mean ball coverage at d = 3 (Table 10) is at least tau, "tighter" otherwise.
  Measured "no tighter" when the ball bound cuts at most 1.05 times the patterns the
  host-graph bound cuts. Reported correct or wrong per dataset.

## T6. Cheap additions

- Table 12: the gSpan time (`gspan_seconds` of `results/gspan_baseline_*.csv`, measurement
  machine) beside the enumerator time.
- Table 7: b_d / |P(K)| per cell, |P(K)| the pool size of the configuration; shown as the range
  over the cells of each row.
- Table 5: a null position beside the two parameter-free selectors' medians. The front
  artefacts store the distinct edge levels of each front, not how many front members sit on
  each level, so the null computed is the expected position of a uniformly chosen **edge
  level** of the front, per cell. The per-member null needs every front re-enumerated
  (Cuneiform K = 6 alone took 6.2 hours) and is not run unless asked.

## Deviations

Recorded as they happened; the results they concern are named.

1. **T5 self-test (before the four-dataset run).** The unsound bound used to show that the
   front comparison catches an unsound bound was first rem = 0. On QM9_bond it cut 49 patterns
   and happened to lose no front member, so it showed nothing. It was replaced by a bound that
   cuts every non-leaf pattern, which the comparison catches. The three real bounds and the
   threshold were not changed.
2. **T6, Table 7 (layout, after seeing the table).** A b_d / |P(K)| column made the table 372.6pt
   wide at the smallest column gap, over the 372pt text width; type size is not reduced. The
   ranges per row went into the table note instead.
3. **T6, Table 5 (observation, not a change).** Every one of the 181 front cells has contiguous
   edge levels, so the pre-registered null over edge levels is 0.5 in every cell by
   construction and carries no information beyond the midpoint. The per-member null still
   needs the fronts re-enumerated with member counts per level.
4. **T4 (post hoc, after seeing the results).** Because the one-edge null matches every
   per_edge cell, the sum-mode cells were also scored against the null "the optimum has the
   pool's largest edge count". This comparison was not pre-registered and is reported as
   post hoc.
