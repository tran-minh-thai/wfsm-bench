@echo off
REM ============================================================================
REM  wfsm-bench - the full experimental suite, on the experiment machine
REM
REM  Every number the paper reports comes from here. Nothing produced on a
REM  development box is quotable, so this reproduces all of it in one place, on
REM  one machine, from one commit.
REM
REM  Stages are ordered by result-per-hour, not by contribution number, so a run
REM  that gets cut short leaves the most useful subset finished. The two size-6
REM  stages are last because a single exhaustive enumeration of 42523 patterns
REM  takes about six and a quarter hours -- measured at 22396.7 s in stage 16 on
REM  2026-08-21, against the two and a half hours this line used to claim -- and
REM  they need several between them.
REM
REM    stage                                supports   approx
REM    1  QM9 rho calibration               C4          15 min
REM    2  QM9 method arm                    C8          30 min
REM    3  Cuneiform degeneracy, sizes 4-5   C2 C3 C4    20 min
REM    4  Cuneiform methods, size 5         C5 C8       25 min
REM    5  anti-monotone pruning A/B         C7           1 h
REM    6  Cuneiform landscape, all 21 cells C1           7 h
REM    7  Cuneiform methods, size 6         C5 C8      5.5 h
REM    8  Cuneiform size 4, loose budget    C8          10 min
REM    9  _bond method arm, 3 collections   C8          ~2 h
REM   10  quantum sweep on QM9              C8        ~1.7 h
REM   11  basin structure, 5 collections    C8         ~30 m
REM   12  alchemy method arm, flag test     C8         ~40 m
REM   13  selector on disjoint seeds        C8        ~2.5 h
REM   14  determinism: stage 13 vs earlier  -          ~10 s
REM   15  calibrate rho, 4 collections      C4         ~15 m
REM   16  Pareto selection on Cuneiform     -          ~6.5 h  not machine-dependent
REM   17  Pareto selection, 5 collections   -          ~2 h    not machine-dependent
REM   18  gSpan head to head, timed         -          ~3 h    MEASURES TIME
REM   19  gSpan agreement, counts only      -          ~7 h    not machine-dependent
REM   20  enumeration cost, third bound   -        ~9-11 h  MEASURES TIME
REM
REM  RHOS was extended below 0.3 on 08-18, so stages 1, 3 and 15 -- the three that
REM  calibrate -- must all be re-run together. Running one leaves the artefacts on
REM  two different sweeps, and the largest stable value would then be comparable
REM  across collections only by luck:  run_experiments_windows.bat 1 3 15
REM                                                    -------
REM                                                    ~19 h
REM
REM  Every stage writes CSV rows as it produces them, so an interrupted stage
REM  keeps what it finished. Re-running a stage overwrites only its own files.
REM
REM  Code reaches this machine by git, never by file sync. Name the branch -- a bare
REM  `git pull` updates whichever one the machine is standing on, which is how a stage
REM  once ran three commits behind. The work is on main since 2026-09-25; a checkout
REM  still on the deleted per-edge-objective moves across as docs\RUN_ON_WINDOWS.md says:
REM
REM      git checkout main
REM      git pull --ff-only
REM      run_experiments_windows.bat            (all stages)
REM      run_experiments_windows.bat 3 4 5      (only some)
REM      run_experiments_windows.bat 18         (gSpan baseline; this one DOES measure
REM                                              time, so it must run here)
REM      run_experiments_windows.bat 16 17      (Pareto selection, Cuneiform then the
REM                                              five _bond collections; neither is in
REM                                              the default
REM                                              list because it measures counts, not
REM                                              time, so any machine gives the same
REM                                              answer)
REM
REM  --ff-only is not decoration. A merge made here would carry a commit hash that
REM  exists on no other machine, so the hash recorded beside the artefacts would name
REM  something nobody else can check out.
REM
REM  Each stage publishes what it wrote as soon as it finishes -- see publish_results.py.
REM  Nothing needs pushing by hand.
REM
REM  Each stage refuses to start if src\ or scripts\ carry uncommitted changes,
REM  because the commit hash written beside its artefacts would then describe code
REM  that never ran. If that refusal fires here, the edit belongs on the development
REM  machine: commit and push there, pull here, run again.
REM ============================================================================
setlocal enabledelayedexpansion

set PY=.venv\Scripts\python.exe
set WORKERS=8
set STAGES=%*
REM 15 runs early although it is numbered last. The number identifies the stage and
REM never changes once artefacts carry it; the order is what the argument needs. Every
REM stage that searches under discounting refuses without a calibrated rho, so the
REM calibration has to precede them.
if "%STAGES%"=="" set STAGES=1 15 2 3 4 5 6 7 8 9 10 11 12 13 14

if not exist "%PY%" (
  echo [error] no virtual environment at %PY%
  echo         Set one up first: run_all_windows.bat, or docs\WINDOWS_STEPS.txt
  exit /b 1
)

echo.
echo ============================================================
echo  wfsm-bench full experimental suite
echo  stages: %STAGES%    workers: %WORKERS%
echo  started %DATE% %TIME%
echo ============================================================

echo.
echo [setup] dependencies, datasets, tests
%PY% -m pip install -q -r requirements.txt || goto :failed
%PY% scripts\prepare_datasets.py || goto :failed
%PY% -m pytest -q || goto :failed
REM Which versions produced these numbers. An experimental section that cannot
REM answer "which numpy?" cannot answer a reviewer who asks.
REM
REM One file per stage, not one for the whole suite. A single shared file records
REM only whichever invocation ran last, so a suite run in pieces over several days
REM ends up claiming that every artefact came from the final piece. Worse, it says
REM nothing at all about the stages that ran under an earlier commit -- and stage 3
REM did exactly that, under a commit whose --suffix handling had not been fixed yet.
REM Inside a parenthesised block %VAR% expands once, when the block is parsed, so a
REM per-iteration %TIME% prints the same clock for every pass -- stage 13 reported five
REM datasets at 21:24:25.47 and none of them started then. !VAR! is read each pass, which
REM is why delayed expansion is on at the top of this file.
for %%S in (%STAGES%) do (
  REM The discount is resolved by the runners themselves, from the calibration artefacts,
REM and a collection without one refuses to run under discounting rather than taking a
REM default. That replaces the flag this file used to pass: two stages forgot it, ran QM9
REM at 0.5 while another used its calibrated 0.6, and searched a different objective for
REM four days. A default that is correct per collection cannot be forgotten.

%PY% scripts\environment.py --require-current --require-clean ^
       --out results\environment_stage%%S.txt || goto :failed
  call :stage%%S
  REM  Publish immediately, not at the end of the suite. The measurement machine is
  REM  not the machine the results are read on, so a stage that finishes without
  REM  pushing has produced nothing anyone else can use -- and every hand-push so
  REM  far went wrong in a different way: one merged where the branch forbids it,
  REM  another pushed the logs and silently left the CSVs behind a stale .gitignore
  REM  glob. Publishing per stage also means an interrupted suite has already
  REM  delivered every stage that completed.
  %PY% scripts\publish_results.py --stage %%S || goto :failed
)
goto :finished

REM --------------------------------------------------------------------------
:stage16
echo.
echo [16] Pareto selection: where a front puts the answer on the size axis   %TIME%
REM  The standing objection to a degenerate scalarised objective is to stop scalarising
REM  and report the non-dominated set instead. A front is a set, though, and a
REM  practitioner wants one pattern, so this measures where the usual selectors land.
REM
REM  NOT a timing measurement. Every column is a count or a pattern size, so the result
REM  is a function of (commit, data) and not of this machine -- it does not need the
REM  designated measurement machine and is not in the default stage list for that
REM  reason. It is here so it runs the same way as everything else when asked for.
REM
REM  One enumeration serves all 21 cells. The pools nest, and tests\test_pool_nesting.py
REM  holds that to a direct mining rather than leaving it as a reading of the source.
REM  Mining each cell separately would re-walk the same space 21 times.
REM
REM  --graph-limit 60 is not a shortcut. Every min_support in weight_mode_landscape.csv
REM  is a fraction of 60 and its graph_limit column says 60, so 60 is what the 21 cells
REM  of the degeneracy result mean. The whole 267 answers a different question under the
REM  same cell names, and costs about four times as much.
REM
REM  Three runs, each writing its own artefact, because the cost is wildly uneven: bound 4
REM  is half a minute, bound 5 a few minutes, and bound 6 about two and a half hours -- the
REM  same enumeration of 42523 patterns that makes stages 6 and 7 the long ones. Splitting
REM  them means an interrupt keeps every cell that finished instead of losing all 21, and
REM  the 14 cells from bounds 4 and 5 are on disk long before the expensive one starts.
REM  Re-mining at bound 5 costs a couple of minutes and buys that.
%PY% scripts\run_pareto_size.py --dataset Cuneiform --graph-limit 60 --sizes 4 ^
     --suffix _Cuneiform_s4 > results\pareto_size_s4.log 2>&1 || goto :failed
type results\pareto_size_s4.log
echo     7 cells done, artefact path verified   %TIME%
%PY% scripts\run_pareto_size.py --dataset Cuneiform --graph-limit 60 --sizes 4 5 ^
     --suffix _Cuneiform_s45 > results\pareto_size_s45.log 2>&1 || goto :failed
type results\pareto_size_s45.log
echo     14 cells done   %TIME%
echo     the next run is the expensive one -- about two and a half hours   %TIME%
%PY% scripts\run_pareto_size.py --dataset Cuneiform --graph-limit 60 --sizes 4 5 6 ^
     --suffix _Cuneiform > results\pareto_size_Cuneiform.log 2>&1 || goto :failed
type results\pareto_size_Cuneiform.log
echo     all 21 cells done %TIME%
exit /b 0

:stage17
echo.
echo [17] Pareto selection on the five _bond collections   %TIME%
REM  Stage 16 answered the question on one collection, and one collection is one
REM  collection whatever the cell count. This repeats it on the five molecular ones.
REM
REM  --graph-limit 200 and the 0.01 support are those collections' own settings, the ones
REM  weight_mode_landscape_*.csv was written under. 0.01 stays in the grid even though the
REM  rest is Cuneiform's, because it is the one cell whose pool can be cross-checked
REM  against an existing artefact -- and a pool cross-check is what caught stage 16
REM  measuring the wrong cells entirely.
REM
REM  THREE bounds, and they are 3-4-5 rather than 4-5-6 on purpose. The question is
REM  whether the answer stabilises as the bound widens, which needs a third point but not
REM  any particular three: two bounds cannot tell 'tracks the bound' apart from 'the
REM  enumeration had not yet reached the right pattern'. Cost decides the rest. Measured
REM  enumeration times: Cuneiform 33.7 s, 524.6 s, 22396.7 s at bounds 4, 5, 6 -- each
REM  step 15x then 43x, and growing. BZR_bond at n=200 is 21.7 s and 133.0 s at bounds 4
REM  and 5, so bound 6 would be over an hour per collection and roughly eight for all
REM  five, against about ten minutes for the whole stage at 3-4-5.
REM
REM  A probe at 3-4-5 on BZR_bond confirms the window is the right one: the scalarised
REM  pick is 2, 3, 4 across the three bounds -- the bound minus one, every time -- while
REM  both parameter-free picks are 2, 2, 3. The contrast the stage exists to measure is
REM  visible inside this range, so widening it buys nothing but hours.
REM
REM  Not a timing measurement: every column is a count or a pattern size. Any machine
REM  gives the same answer, which is why this is not in the default stage list.
for %%D in (BZR_bond COX2_bond DHFR_bond QM9_bond alchemy_full_bond) do (
  echo     %%D   !TIME!
  %PY% scripts\run_pareto_size.py --dataset %%D --graph-limit 200 --sizes 3 4 5 ^
       --supports 0.01 0.0333333 0.10 0.20 0.30 0.50 0.60 0.70 ^
       --suffix _%%D > results\pareto_size_%%D.log 2>&1 || goto :failed
  type results\pareto_size_%%D.log
)
echo     done %TIME%
exit /b 0

:stage18
echo.
echo [18] gSpan as a published baseline, head to head with this project's enumerator   %TIME%
REM  gSpan and this project's enumerator answer the same question -- every frequent
REM  connected subgraph up to a size bound -- so their cost is comparable. The
REM  metaheuristics do not: they search for one best pattern rather than enumerating, so
REM  timing them against gSpan would compare two different problems.
REM
REM  THIS ONE DOES MEASURE TIME, so unlike stages 16 and 17 it belongs on this machine and
REM  nowhere else. Both sides are timed inside one process on one cell; a speed-up
REM  assembled from two runs would be arithmetic across different caches and heaps.
REM
REM  Bound 6 runs gSpan alone. The enumerator needs about six hours there -- measured at
REM  22396.7 s in stage 16 -- and its column is left EMPTY rather than filled from that
REM  log, because a number lifted out of another run is not a comparison.
REM
REM  Cost, extrapolated from a development-machine probe and therefore an estimate: the
REM  enumerator dominates, at roughly half an hour per collection at bound 5.
for %%D in (Cuneiform) do (
  echo     %%D  bounds 3-4-5, both sides   !TIME!
  %PY% scripts\run_gspan_baseline.py --dataset %%D --graph-limit 60 --sizes 3 4 5 ^
       --support 2 --suffix _%%D > results\gspan_%%D.log 2>&1 || goto :failed
  type results\gspan_%%D.log
)
for %%D in (BZR_bond COX2_bond DHFR_bond QM9_bond alchemy_full_bond) do (
  echo     %%D  bounds 4-5, both sides   !TIME!
  %PY% scripts\run_gspan_baseline.py --dataset %%D --graph-limit 200 --sizes 4 5 ^
       --support 2 --suffix _%%D > results\gspan_%%D.log 2>&1 || goto :failed
  type results\gspan_%%D.log
)
echo     head-to-head done; the next run is gSpan alone at bound 6   %TIME%
%PY% scripts\run_gspan_baseline.py --dataset Cuneiform --graph-limit 60 --sizes 6 ^
     --support 2 --skip-exhaustive --suffix _Cuneiform_s6 > results\gspan_Cuneiform_s6.log 2>&1 || goto :failed
type results\gspan_Cuneiform_s6.log
echo     done %TIME%
exit /b 0

:stage20
echo.
echo [20] Enumeration cost: the third size bound, so each collection is a curve   %TIME%
REM  The cost table reads the timed enumeration rows written by stage 18. Five collections
REM  have two bounds there and Cuneiform has three, and two points do not make a curve --
REM  the figure of enumeration cost has been unbuildable for that reason across two rounds
REM  of briefs. One more bound on each collection is what closes it.
REM
REM  THIS MEASURES TIME, so it belongs on this machine and nowhere else. Both sides are timed
REM  inside one process on one cell; a speed-up assembled from two runs would be arithmetic
REM  across different caches and heaps.
REM
REM  The writer carries over the bounds already timed, so naming bound 6 alone extends each
REM  file instead of replacing it. That was NOT true before 2026-09-11: the timed writer
REM  truncated first, exactly as the counts-only writer used to, and this command would have
REM  destroyed several hours of seconds before starting. Seconds are worse to lose than
REM  counts, because they can be recomputed only here.
REM
REM  Cost. Cuneiform is ANCHORED: that exact enumeration was measured at 22396.7 s in stage
REM  16 and gSpan's side of the cell at 517.95 s, so about six and a half hours. The five
REM  _bond collections are EXTRAPOLATED from the table's own ratios -- pool x2.5-2.8 and
REM  seconds per pattern x2.2-2.5 for each edge added -- giving roughly 40 to 60 minutes
REM  each. Nine to eleven hours in total, and the extrapolated half could be wrong by a
REM  factor; it has never been run at this bound.
for %%D in (BZR_bond COX2_bond DHFR_bond QM9_bond alchemy_full_bond) do (
  echo     %%D  bound 6, both sides timed   !TIME!
  %PY% scripts\run_gspan_baseline.py --dataset %%D --graph-limit 200 --sizes 6 ^
       --support 2 --suffix _%%D >> results\gspan_%%D.log 2>&1 || goto :failed
  type results\gspan_%%D.log
)
echo     the five smaller collections are done; Cuneiform is the long one   %TIME%
%PY% scripts\run_gspan_baseline.py --dataset Cuneiform --graph-limit 60 --sizes 6 ^
     --support 2 --suffix _Cuneiform >> results\gspan_Cuneiform.log 2>&1 || goto :failed
type results\gspan_Cuneiform.log
echo     done %TIME%
exit /b 0

:stage19
echo.
echo [19] gSpan agreement: do the two pattern SETS match, over how many patterns   %TIME%
REM  Stage 18 measures time and so belongs on this machine. This one measures neither
REM  side's seconds -- every column is a count or a set comparison, so the result is a
REM  function of (commit, data) and not of the machine. It writes to results-invariant\,
REM  which carries no timing column at all, and it runs here only so it runs the same way
REM  as everything else when asked for.
REM
REM  The enumerator is NEVER skipped here. With nothing to compare against, the file
REM  would record a count and call it an agreement.
REM
REM  Why only Cuneiform bound 6. Twenty of the twenty-one cells are already on disk and
REM  under version control; the writer carries over every cell this run does not
REM  recompute, so naming bound 6 alone extends the file instead of replacing it. That
REM  was not true before 2026-09-04: the writer truncated first, and this command would
REM  have destroyed the two finished bounds before starting. Widen --sizes to recompute
REM  more; each named bound replaces its own row and never doubles it.
REM
REM  Cost, ANCHORED not extrapolated: this exact enumeration -- Cuneiform n=60, bound 6,
REM  support 2, 42523 patterns -- was measured on this machine at 22396.7 s in stage 16,
REM  logged in results\pareto_size_Cuneiform.log. gSpan's side of the same cell took
REM  517.95 s. Allow about seven hours: the comparison of the two pattern sets is on top
REM  of both, and has never been timed separately.
%PY% scripts\run_gspan_baseline.py --dataset Cuneiform --graph-limit 60 --sizes 6 ^
     --support 2 --counts-only --suffix _full_Cuneiform > results\gspan_agree_Cuneiform_s6.log 2>&1 || goto :failed
type results\gspan_agree_Cuneiform_s6.log
echo     done %TIME%
exit /b 0

:stage1
echo.
echo [1] QM9 rho calibration   %TIME%
REM QM9 cannot borrow Cuneiform's rho: where the optimum leaves the size bound
REM depends on how fast support falls with pattern size, a property of the data.
%PY% scripts\run_degeneracy.py --dataset QM9_bond --limit 200 --sizes 4,5 ^
     --supports 2 --calibrate-rho --suffix _qm9 > results\qm9_rho.log 2>&1 || goto :failed
echo     done %TIME%
exit /b 0

:stage2
echo.
echo [2] QM9 method arm   %TIME%
REM --dataset is what makes this QM9. Without it the run silently uses the
REM preset's dataset, which is exactly how a stage labelled _qm9 once produced
REM Cuneiform numbers -- and nothing said so until the ranks came out identical.
call :modes_for QM9_bond
%PY% scripts\run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 ^
     --sizes 5 --supports 2 --modes !MODES! --seeds 20 ^
     --workers %WORKERS% ^
     --suffix _qm9 > results\qm9_methods.log 2>&1 || goto :failed
%PY% scripts\run_statistics.py --runs results\weight_mode_runs_qm9.csv ^
     > results\qm9_statistics.log 2>&1
echo     done %TIME%
exit /b 0

:stage3
echo.
echo [3] Cuneiform degeneracy, sizes 4-5   %TIME%
REM The factorization, the two step costs, the gamma sweep that rules out fixing
REM the degeneracy with the penalty coefficients, and the rho calibration.
%PY% scripts\run_degeneracy.py --dataset Cuneiform --limit 60 --sizes 4,5 ^
     --calibrate-rho --suffix _cuneiform > results\cuneiform_degeneracy.log 2>&1 || goto :failed
echo     done %TIME%
exit /b 0

:stage4
echo.
echo [4] Cuneiform methods, size 5   %TIME%
call :modes_for Cuneiform
%PY% scripts\run_weight_mode.py --preset deep --sizes 5 --supports 2 ^
     --modes !MODES! --seeds 20 --workers %WORKERS% ^
     --suffix _s5 > results\s5_methods.log 2>&1 || goto :failed
%PY% scripts\run_statistics.py --runs results\weight_mode_runs_s5.csv ^
     > results\s5_statistics.log 2>&1
echo     done %TIME%
exit /b 0

:stage5
echo.
echo [5] anti-monotone pruning A/B   %TIME%
REM Same cell, one condition apart. The threshold binds here, which is where the
REM wasted budget shows up at all.
%PY% scripts\run_weight_mode.py --preset deep --sizes 6 --supports 30 ^
     --modes sum --seeds 20 --workers %WORKERS% ^
     --suffix _noprune > results\noprune.log 2>&1 || goto :failed
%PY% scripts\run_weight_mode.py --preset deep --sizes 6 --supports 30 ^
     --modes sum --seeds 20 --workers %WORKERS% --prune ^
     --suffix _prune > results\prune.log 2>&1 || goto :failed
echo     done %TIME%
exit /b 0

:stage6
echo.
echo [6] Cuneiform landscape, all 21 cells   %TIME%
REM sum and per_edge share one enumeration per cell -- per_edge divides in the
REM fitness, so the Evaluation is the same. This is the evidence for C1.
%PY% scripts\run_weight_mode.py --preset deep --supports 2,6,12,18,30,36,42 ^
     --modes sum,per_edge --landscape-only ^
     > results\landscape_sweep.log 2>&1 || goto :failed
echo     done %TIME%
exit /b 0

:stage7
echo.
echo [7] Cuneiform methods, size 6   %TIME%
call :modes_for Cuneiform
%PY% scripts\run_weight_mode.py --preset deep --sizes 6 --supports 2 ^
     --modes !MODES! --seeds 20 --workers %WORKERS% ^
     --suffix _s6 > results\s6_methods.log 2>&1 || goto :failed
%PY% scripts\run_statistics.py --runs results\weight_mode_runs_s6.csv ^
     > results\s6_statistics.log 2>&1
echo     done %TIME%
exit /b 0

:stage8
echo.
echo [8] Cuneiform size 4, loose budget   %TIME%
REM Separates C8's two hypotheses. On QM9 the ranking reversed -- ILS first on
REM Cuneiform, sixth of seven there -- but the two runs differed in two ways at
REM once: local-optimum density (2.1%% against 11.9%%) and budget-to-space ratio
REM (1.2-4.8%% against 9.8-39.2%%). Two points fit any line.
REM
REM Cuneiform at size 4 has a pool of 712, near QM9's 613, so the same budgets give
REM 8.4-33.7%% against QM9's 9.8-39.2%%. That holds the ratio and changes only the
REM dataset. If ILS stays first here the ratio is not the cause and the landscape
REM is; if it drops, the ratio is.
REM
REM Shrinking QM9's budgets instead would have been wrong: 6 to 24 evaluations is
REM under one generation for a swarm of 20, which measures initialisation rather
REM than search -- the same reason budget 30 was dropped from the grid.
call :modes_for Cuneiform
%PY% scripts\run_weight_mode.py --preset deep --sizes 4 --supports 2 ^
     --modes !MODES! --seeds 20 --workers %WORKERS% ^
     --suffix _s4loose > results\s4loose.log 2>&1 || goto :failed
%PY% scripts\run_statistics.py --runs results\weight_mode_runs_s4loose.csv ^
     > results\s4loose_statistics.log 2>&1
echo     done %TIME%
exit /b 0

:stage9
echo.
echo [9] _bond method arm, 3 collections   %TIME%
REM Three explanations for the ranking reversal are ruled out; what was left was
REM fitness_levels, 0.9967 on QM9 against 0.287 on Cuneiform, confounded with graph
REM structure. BZR_bond, COX2_bond and DHFR_bond were meant to separate the two.
REM They do not: probing them gives fitness_levels of 1.00 at both sizes, so they
REM sit beside QM9 on that axis as well as on structure. The separation has to come
REM from somewhere else.
REM
REM They are still worth running, for one thing only. All three are QM9-like on both
REM axes, so if iterated local search comes last on them the result is consistent
REM with either explanation and settles nothing -- but if it comes first, both are
REM refuted at once. A test that can only refute is still a test; a test whose
REM confirming outcome would mean nothing is what this must not be mistaken for.
REM
REM The landscape half was already measured off-machine and is a property of the
REM data, so this runs the method arm. Budgets stay at the QM9 values rather than
REM being rescaled to the smaller pool: the ratio was shown to change the size of
REM the difference and not its direction, while rescaling would put the smallest
REM budget under two generations of a twenty-particle swarm, which measures
REM initialisation instead of search.
for %%D in (BZR COX2 DHFR) do (
  call :modes_for %%D_bond
  echo     %%D_bond   modes=!MODES!   !TIME!
  %PY% scripts\run_weight_mode.py --preset deep --dataset %%D_bond --limit 200 ^
       --sizes 5 --supports 2 --modes !MODES! --seeds 20 ^
       --workers %WORKERS% ^
       --suffix _%%Dbond > results\%%Dbond_methods.log 2>&1 || goto :failed
  %PY% scripts\run_statistics.py --runs results\weight_mode_runs_%%Dbond.csv ^
       > results\%%Dbond_statistics.log 2>&1
)
echo     done %TIME%
exit /b 0

:stage10
echo.
echo [10] quantum sweep on QM9   %TIME%
REM  The controlled experiment. Every dataset found so far ties plateau structure to
REM  graph structure -- Cuneiform has rings and heavy plateaus, the molecular sets have
REM  trees and no ties at all -- so no collection can separate them. --quantum rounds
REM  the objective the search navigates onto a grid, leaving the true objective for
REM  reporting, which varies plateau structure on a fixed dataset with the graphs, the
REM  pool, the budgets and the seeds all held where they are.
REM
REM  REGISTERED BEFORE RUNNING, so the sweep cannot be read backwards:
REM    Prediction. If plateau structure is what iterated local search needs, its mean
REM    rank improves as the quantum grows. It sits sixth of seven on QM9 untouched.
REM    Refutation. A flat or worsening rank across the sweep kills the explanation,
REM    and the last variable that survived the earlier eliminations goes with it.
REM    Also watched, not predicted: whether ACO keeps its lead as ties multiply.
REM
REM  The grid was measured, not guessed. On this pool of 613 the quanta below give
REM  fitness_levels of 0.997, 0.692, 0.475, 0.310, 0.188, 0.087, 0.047 -- and 0.310 is
REM  where Cuneiform sits at 0.287, so the sweep spans the natural range and passes
REM  through the point of interest rather than around it.
REM
REM  ONE HONEST LIMIT. Quantising raises the local-optimum count instead of lowering
REM  it: equal neighbours mean no strictly fitter neighbour, so plateaus manufacture
REM  local optima. Cuneiform pairs low levels with SPARSE local optima (2.1%), while a
REM  quantised QM9 pairs low levels with dense ones (31% and rising). So this varies
REM  plateau density, and does not reproduce Cuneiform. Both columns are written --
REM  fitness_levels against fitness_levels_true, local_optima against local_optima_true
REM  -- so the size of the manipulation is measured in every row rather than inferred
REM  from the grid width.
REM
REM  sum only. discounted would change the objective as well, confounding the one
REM  variable this stage exists to isolate -- and the rank being explained was measured
REM  under sum.
call :quantum q000 0        || goto :failed
call :quantum q002 0.0002   || goto :failed
call :quantum q005 0.0005   || goto :failed
call :quantum q010 0.001    || goto :failed
call :quantum q020 0.002    || goto :failed
call :quantum q050 0.005    || goto :failed
call :quantum q100 0.01     || goto :failed
echo     done %TIME%
exit /b 0

:quantum
echo     quantum %2   !TIME!
%PY% scripts\run_weight_mode.py --preset deep --dataset QM9_bond --limit 200 ^
     --sizes 5 --supports 2 --modes sum --seeds 20 --workers %WORKERS% ^
     --quantum %2 --suffix _%1 > results\%1_methods.log 2>&1 || exit /b 1
%PY% scripts\run_statistics.py --runs results\weight_mode_runs_%1.csv ^
     > results\%1_statistics.log 2>&1
exit /b 0

:stage11
echo.
echo [11] basin structure, 5 collections   %TIME%
REM  The first instance-level quantity that lines up with the ranking, after every
REM  aggregate one failed. From each pattern in the pool, steepest ascent along
REM  single-edit moves, recording where it stops; the share of the pool that ends on
REM  an optimal pattern is how findable the answer is by local improvement.
REM
REM  Probed off-machine: the optimum's basin holds 14 to 16 percent on Cuneiform and
REM  COX2, where ACO ranks fifth and fourth, and 0.16 to 0.56 percent on BZR, DHFR and
REM  QM9, where it ranks first on all three. Optimum multiplicity is one everywhere, so
REM  no basin is wide merely because many patterns tie for best.
REM
REM  Deterministic and RNG-free, so these are data properties rather than performance
REM  numbers -- but the paper quotes this machine, so it runs here.
REM
REM  Cuneiform is last: its enumeration is the expensive one, and an interrupted stage
REM  should leave the four cheap collections finished.
call :modes_for QM9_bond
%PY% scripts\run_basins.py --dataset QM9_bond  --limit 200 --sizes 5 --supports 2 ^
     --modes !MODES! ^
     --suffix _qm9 > results\basins_qm9.log 2>&1 || goto :failed
REM alchemy carries the numbers that the stage-12 prediction was registered against, so
REM its basins have to exist here as an artefact rather than only in a probe that was
REM deleted -- a predictor quoted in the log with no file behind it is a dangling claim.
call :modes_for alchemy_full_bond
%PY% scripts\run_basins.py --dataset alchemy_full_bond --limit 200 --sizes 5 --supports 2 ^
     --modes !MODES! --suffix _alch > results\basins_alch.log 2>&1 || goto :failed
for %%D in (BZR COX2 DHFR) do (
  call :modes_for %%D_bond
  %PY% scripts\run_basins.py --dataset %%D_bond --limit 200 --sizes 5 --supports 2 ^
       --modes !MODES! --suffix _%%Dbond > results\basins_%%Dbond.log 2>&1 || goto :failed
)
call :modes_for Cuneiform
%PY% scripts\run_basins.py --dataset Cuneiform --limit 60 --sizes 5 --supports 2 ^
     --modes !MODES! --suffix _cuneiform > results\basins_cuneiform.log 2>&1 || goto :failed
echo     done %TIME%
exit /b 0

:stage12
echo.
echo [12] alchemy method arm, flag test   %TIME%
REM  The registered test of the binary flag, on a collection that took no part in
REM  forming it. alchemy_full is the only TUDataset collection besides Cuneiform and
REM  QM9 carrying published bond types together with geometry, so its edge labels are
REM  read rather than assigned constant.
REM
REM  It gives both flag values on one set of graphs. Under sum the optimum's basin
REM  holds 0.47 percent while something else drains 51.2, so the flag is off; under
REM  discounted the optimum drains 78.5 percent and is the largest attractor, so the
REM  flag is on. Same graphs, same pool of 428, only the objective differs -- a
REM  within-dataset comparison, which is the evidence the continuous measure failed on.
REM
REM  PREDICTIONS, WRITTEN BEFORE THIS RAN, flag measured and ranking unknown:
REM    (1) directional, the main one: ACO ranks better under sum than under discounted.
REM        Prior 50 percent.
REM    (2) absolute, stronger: under sum, ACO comes first of seven. Prior about 1 in 7.
REM    (3) weak, recorded for completeness: under discounted it does not come first.
REM        Prior 6 in 7, so it proves little alone.
REM  Either (1) or (2) failing refutes the flag.
REM
REM  WHAT WOULD VOID THE TEST, also written first: a basin of 78.5 percent over nine
REM  attractors is a very smooth landscape, and if every method reaches the optimum then
REM  every pair ties, n_effective collapses and the ranking is noise. If Friedman under
REM  discounted is not significant, or median n_effective is under 5, that cell is
REM  UNTESTED rather than passed.
call :modes_for alchemy_full_bond
%PY% scripts\run_weight_mode.py --preset deep --dataset alchemy_full_bond --limit 200 ^
     --sizes 5 --supports 2 --modes !MODES! --seeds 20 --workers %WORKERS% ^
     --suffix _alch > results\alch_methods.log 2>&1 || goto :failed
%PY% scripts\run_statistics.py --runs results\weight_mode_runs_alch.csv ^
     > results\alch_statistics.log 2>&1
echo     done %TIME%
exit /b 0

:stage13
echo.
echo [13] selector on disjoint seeds   %TIME%
REM  Racing the methods at the cheap budget and keeping the winner beat every fixed
REM  rule -- 0.25 percent mean regret against 1.55 for always taking ACO, the best fixed
REM  rule these collections offer. But that measurement raced and paid on the same
REM  seeds, so the race saw the very runs it was judged on, and the figure is optimistic
REM  by an amount nobody knows. Unknown is what makes it unusable: a selector reported
REM  at 0.25 that is really 1.5 is a different recommendation.
REM
REM  Forty seeds per cell, raced on the even ones and paid on the odd, so no run appears
REM  on both sides. Seeds 0 to 19 also reproduce the existing twenty-seed artefacts,
REM  which makes this a determinism check at no extra cost.
REM
REM  Budgets are cut to the two the selector actually uses. The race reads the cheapest
REM  and the bill is paid at the dearest, so 120 and 180 would be search time bought and
REM  never read -- and dropping them keeps two budgets by forty seeds at eighty blocks
REM  per Friedman, the same as the four-by-twenty grids everywhere else.
REM
REM  WHAT WOULD OVERTURN THIS, written first: if racing's regret on disjoint seeds rises
REM  to meet the best fixed rule, the recommendation collapses to "always take ACO" and
REM  the selector is not worth its cost. run_selector.py prints that verdict either way
REM  rather than only when it is favourable.
REM Cuneiform reads sixty graphs and the rest read two hundred. That is not a
REM preference: every artefact for Cuneiform -- its ranking, its landscape, its basins --
REM was measured at sixty, and reading two hundred here would build a different instance
REM and compare it against numbers that describe another one. So it gets its own call
REM rather than riding the loop.
call :modes_for Cuneiform
%PY% scripts\run_weight_mode.py --preset deep --dataset Cuneiform ^
     --limit 60 --sizes 5 --supports 2 --modes !MODES! ^
     --budgets 60,240 --seeds 40 --workers %WORKERS% ^
     --suffix _selCuneiform > results\selCuneiform_methods.log 2>&1 || goto :failed
REM QM9 leaves the loop for the same reason Cuneiform did: it needs an argument the
REM others do not. The remaining four have never been calibrated and take the default,
REM which is consistent across every stage that touches them but is a gap of its own.
call :modes_for QM9_bond
%PY% scripts\run_weight_mode.py --preset deep --dataset QM9_bond ^
     --limit 200 --sizes 5 --supports 2 --modes !MODES! ^
     --budgets 60,240 --seeds 40 --workers %WORKERS% ^
     --suffix _selQM9_bond > results\selQM9_bond_methods.log 2>&1 || goto :failed
for %%D in (BZR_bond COX2_bond DHFR_bond alchemy_full_bond) do (
  call :modes_for %%D
  echo     %%D   modes=!MODES!   !TIME!
  %PY% scripts\run_weight_mode.py --preset deep --dataset %%D ^
       --limit 200 --sizes 5 --supports 2 --modes !MODES! ^
       --budgets 60,240 --seeds 40 --workers %WORKERS% ^
       --suffix _sel%%D > results\sel%%D_methods.log 2>&1 || goto :failed
)
%PY% scripts\run_selector.py --runs "results/weight_mode_runs_sel*.csv" ^
     > results\selector.log 2>&1 || goto :failed
type results\selector.log
REM  The all-pairs table the selector's regret is quoted from. Its inputs are fixed in
REM  run_selector.py (ALL_PAIRS_RUNS), so this line cannot drift from what the table needs.
%PY% scripts\run_selector.py --all-pairs --out results\selector_regret_allpairs.csv ^
     > results\selector_allpairs.log 2>&1 || goto :failed
echo     done %TIME%
exit /b 0

:stage14
echo.
echo [14] determinism: stage 13 against the earlier runs   !TIME!
REM  Stage 13 ran forty seeds where the earlier stages ran twenty, so seeds 0 to 19 are
REM  the same searches twice -- once under the commit of mid-August and once under this
REM  one, with --offset, fitness_quantum, a required --suffix and the commit gate added
REM  in between. If any of those moved a number, this is where it shows.
REM
REM  Per-seed rows only. The aggregate families average over different seed counts and
REM  budget grids, so comparing them would report arithmetic as drift.
REM
REM  Seconds, not hours. It is a stage rather than a command to retype because a check
REM  that lives in a chat message is a check that gets skipped.
REM The log is deleted first rather than appended to. A previous pass left behind is a
REM previous pass being read as this one, and the whole point here is knowing which run
REM a number came from.
set DRIFT=0
del /q results\determinism_stage14.log 2>nul
call :determinism selQM9_bond          qm9
call :determinism selBZR_bond          BZRbond
call :determinism selCOX2_bond         COX2bond
call :determinism selDHFR_bond         DHFRbond
call :determinism selalchemy_full_bond alch
call :determinism selCuneiform         s5
if not "!DRIFT!"=="0" (
  echo.
  echo     [failed] !DRIFT! comparison^(s^) disagree. The code changed a number since
  echo              mid-August, or a stage passed different arguments. Check the
  echo              arguments first: the one drift this has caught was a missing --rho
  echo              on QM9, not a change in the scoring code.
  goto :failed
)
type results\determinism_stage14.log
echo     done !TIME!
exit /b 0

:determinism
REM Redirected, not printed. The first version of this stage wrote its verdict to the
REM console and saved nothing, so the run left an environment file recording that it
REM happened and no record of what it found -- which is the same as not having run.
echo.>> results\determinism_stage14.log
echo === %1 vs %2 ===>> results\determinism_stage14.log
%PY% scripts\check_determinism.py --a %1 --b %2 --families weight_mode_runs ^
     >> results\determinism_stage14.log 2>&1
if errorlevel 1 set /a DRIFT+=1
exit /b 0

:stage15
echo.
echo [15] calibrate rho on the four uncalibrated collections   !TIME!
REM  Only Cuneiform and QM9 were ever calibrated, at 0.5 and 0.6. The other four ran
REM  discounting at the default, which is Cuneiform's value and is right for them only
REM  by coincidence. The contributions log says the discount is a per-collection
REM  quantity, chosen by whether the optimum stays interior and on the same pattern as
REM  the size bound widens, and that depends on how fast support falls with pattern
REM  size -- a property of the data, not a constant.
REM
REM  Cheap because the pool is enumerated once per size and rescored for each rho, so
REM  the sweep costs one enumeration rather than one per value.
REM
REM  A collection with no stable rho is a result, not a failure: it would mean no
REM  discount in the swept range gives it an interior optimum that survives a widening
REM  bound, and the runners will then refuse to search it under discounting, which is
REM  the correct outcome rather than a blocked stage.
for %%D in (BZR_bond COX2_bond DHFR_bond alchemy_full_bond) do (
  echo     %%D   !TIME!
  %PY% scripts\run_degeneracy.py --dataset %%D --limit 200 --sizes 4,5 ^
       --supports 2 --calibrate-rho --suffix _%%D > results\rho_%%D.log 2>&1 || goto :failed
)
%PY% scripts\pick_stable_rho.py --report > results\rho_summary.txt 2>&1
type results\rho_summary.txt
echo     done !TIME!
exit /b 0

:modes_for
REM Sets MODES for one collection: both objectives where a discount has been calibrated,
REM sum alone where none has. Without this a collection with no stable rho makes the
REM runner refuse -- correctly -- and the refusal takes the whole stage down with it via
REM the error check. DHFR is that collection: thirteen discount values from 1.0 to 0.05
REM and its peak never stops tracking the size bound, so it has no discounted arm to run.
REM Dropping out of that arm is the right outcome; aborting the stage is not.
set MODES=sum
for /f "usebackq delims=" %%R in (`%PY% scripts\pick_stable_rho.py --dataset %1`) do set MODES=sum,discounted
exit /b 0

REM --------------------------------------------------------------------------
:finished
echo.
echo ============================================================
echo  finished %DATE% %TIME%
echo.
echo  Tables:   %PY% scripts\make_tables.py --out results\tables.md
echo  Figures:  %PY% scripts\make_figures.py
echo ============================================================
exit /b 0

:failed
echo.
echo [failed] a stage returned an error - see the log it was writing.
echo          If the message was [refused], this checkout is not the pushed code:
echo          either it carries uncommitted edits, or it is behind the branch.
echo          Commit and push on the development machine, then here:
echo              git checkout ^<branch^> ^&^& git pull --ff-only
exit /b 1
