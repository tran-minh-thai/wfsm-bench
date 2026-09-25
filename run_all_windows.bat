@echo off
REM ============================================================================
REM  wfsm-bench - full run on Windows
REM
REM  Sets the project up and runs every experiment, writing one log per step to
REM  results\logs. Safe to re-run: it updates an existing checkout instead of
REM  cloning again, and reuses the virtual environment.
REM
REM  Run it either from inside a checkout, or from an empty folder to have it
REM  clone one. Double-click, or from a Command Prompt:  run_all_windows.bat
REM ============================================================================
setlocal enabledelayedexpansion

set REPO_URL=https://github.com/tran-minh-thai/wfsm-bench.git
REM The work is on main since 2026-09-25, when per-edge-objective became main and was
REM deleted. The branch is still named, because a checkout keeps whatever branch it
REM stood on, and :update_checkout below moves an old one across.
set BRANCH=main
set WORKERS=10
set PRESET=balanced

echo.
echo ============================================================
echo  wfsm-bench   preset=%PRESET%   workers=%WORKERS%
echo ============================================================

REM --- Step 0: Python -------------------------------------------------------
echo.
echo [0/6] Checking Python...
python --version 2>NUL
if errorlevel 1 (
    echo   ERROR: python was not found on PATH.
    echo   Install Python 3.11 or newer from python.org, tick "Add to PATH",
    echo   then open a new Command Prompt and run this again.
    goto :fail
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Found Python !PYVER!  ^(3.11 or newer is required^)

REM --- Step 1: get the code -------------------------------------------------
echo.
echo [1/6] Getting the code...
if exist "src\wfsmbench" (
    echo   Already inside a checkout; switching to %BRANCH% and pulling.
    call :update_checkout
    if errorlevel 1 goto :fail
) else (
    if exist "wfsm-bench\src\wfsmbench" (
        echo   Found wfsm-bench\ next to this file; entering and pulling.
        cd wfsm-bench
        call :update_checkout
        if errorlevel 1 goto :fail
    ) else (
        echo   Cloning %REPO_URL% ^(branch %BRANCH%^)
        git clone -b %BRANCH% %REPO_URL%
        if errorlevel 1 (
            echo   ERROR: clone failed. Check the network, and that this account
            echo   has access to the repository ^(it is private^).
            goto :fail
        )
        cd wfsm-bench
    )
)
echo   Working in: %CD%

REM --- Step 2: virtual environment ------------------------------------------
echo.
echo [2/6] Preparing the virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo   Reusing the existing .venv
) else (
    python -m venv .venv
    if errorlevel 1 goto :fail
    echo   Created .venv
)
set PY=.venv\Scripts\python.exe

REM --- Step 3: dependencies -------------------------------------------------
echo.
echo [3/6] Installing dependencies...
if not exist "results\logs" mkdir "results\logs"
%PY% -m pip install --upgrade pip > results\logs\00_pip_upgrade.log 2>&1
%PY% -m pip install -r requirements.txt > results\logs\01_pip_install.log 2>&1
if errorlevel 1 (
    echo   ERROR: install failed. See results\logs\01_pip_install.log
    goto :fail
)
%PY% -c "import networkx, numpy, yaml, matplotlib, pytest; print('networkx', networkx.__version__, '| numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
if errorlevel 1 (
    echo   ERROR: a dependency is still missing. See results\logs\01_pip_install.log
    goto :fail
)

REM --- Step 4: datasets -----------------------------------------------------
echo.
echo [4/6] Preparing the datasets...
if exist "data\BZR_MD.json" if exist "data\Cuneiform.json" (
    echo   Both datasets are already present; skipping the download.
    goto :datasets_ready
)
echo   Downloading and converting from TUDataset ^(about 10 MB, needs internet^)...
%PY% scripts\prepare_datasets.py > results\logs\02_datasets.log 2>&1
if errorlevel 1 (
    echo   ERROR: dataset preparation failed. See results\logs\02_datasets.log
    echo   Check the internet connection: the files come from
    echo   https://www.chrsmrrs.com/graphkerneldatasets
    goto :fail
)
:datasets_ready
dir /b data\*.json

REM --- Step 5: tests --------------------------------------------------------
REM  After the datasets, not before: the tests that exercise the real datasets skip
REM  without them. No expected count is printed -- the one that was, 155, stayed on
REM  screen long after the suite passed a thousand. pytest's own last line is shown.
echo.
echo [5/6] Running the test suite ^(any failure stops the run; skips are expected^)...
%PY% -m pytest -q > results\logs\03_tests.log 2>&1
if errorlevel 1 (
    echo   ERROR: tests failed. See results\logs\03_tests.log
    goto :fail
)
findstr /C:"passed" results\logs\03_tests.log

REM --- Step 6: the experiments ----------------------------------------------
REM  This used to run run_budget_regimes, run_pareto, run_diversity_sweep,
REM  run_full_scale and run_timing. Those predate two measurement fixes -- the
REM  fitness was anchored on the search bound instead of a study constant, and the
REM  reported best ignored the support threshold -- so everything they wrote came
REM  from code now known to be wrong, and their output files were deleted for that
REM  reason. Leaving them wired into the one script a new machine is pointed at
REM  would quietly recreate the whole set.
REM
REM  The current suite is run_experiments_windows.bat: it gates every stage on a
REM  checkout matching its commit, and writes an environment file per stage.
echo.
echo [6/6] Running the experiment suite...
echo       Stages and their timings are listed at the top of run_experiments_windows.bat.
call run_experiments_windows.bat
if errorlevel 1 goto :fail

REM --- Done -----------------------------------------------------------------
echo.
echo ============================================================
echo  FINISHED
echo ============================================================
echo.
echo  Result tables in results\ :
dir /b results\*.csv 2>NUL
echo.
REM  Figures are PDF since 2026-09-13. Nothing in this run draws them -- the suite only
REM  prints the make_figures command -- so whatever is listed may predate this run, and
REM  git ignores the directory, so none of it is pushed. Saying so is the point: a list
REM  under "FINISHED" reads as this run's output.
echo  Figures in results\figures\ ^(drawn by scripts\make_figures.py, NOT by this run,
echo  and not pushed: git ignores them^):
dir /b results\figures\*.pdf 2>NUL
if errorlevel 1 echo    ^(none^)
echo.
echo  Step logs in results\logs\ ^(not pushed: git ignores them^):
dir /b results\logs\*.log 2>NUL
echo.
REM  Not :fail on a non-zero return: every step DID finish, and :fail says otherwise.
call :check_published
set PUBLISHED=%errorlevel%
echo.
pause
exit /b %PUBLISHED%

:check_published
REM  Each stage pushes its own artefacts as it finishes (publish_results.py), so nothing
REM  is sent by hand. This checks that claim instead of printing it. Two things can be
REM  left behind: artefacts never committed, and a commit whose push failed. The next
REM  publish pushes the second kind too (since 2026-09-25; before, it printed [nothing]
REM  and left the commit here), but only if one runs -- a failure in the last stage
REM  leaves nothing after it to do so.
echo  Checking that every artefact reached GitHub...
git fetch --quiet origin
if errorlevel 1 (
    echo    WARNING: could not reach GitHub, so this is UNVERIFIED -- not the same as pushed.
    echo    Run this file again when the network is back.
    exit /b 0
)
REM  Without an upstream the count below fails, stays at zero, and would read as pushed.
git rev-parse --abbrev-ref "@{upstream}" >NUL 2>&1
if errorlevel 1 (
    echo    UNVERIFIED: this branch tracks no branch on GitHub, so nothing can be compared.
    echo    Set it with:  git branch --set-upstream-to=origin/%BRANCH%
    exit /b 1
)
set LEFT_FILES=0
for /f "delims=" %%L in ('git status --porcelain -- "results*"') do set /a LEFT_FILES+=1
set LEFT_COMMITS=0
for /f %%N in ('git rev-list --count "@{upstream}..HEAD"') do set LEFT_COMMITS=%%N
if "%LEFT_FILES%%LEFT_COMMITS%"=="00" (
    echo    All artefacts are on GitHub. Nothing to send by hand.
    exit /b 0
)
echo    NOT everything reached GitHub:
if not "%LEFT_FILES%"=="0" (
    echo      %LEFT_FILES% artefact file^(s^) not committed:
    git status --short -- "results*"
    echo      Publish them with the stage number they belong to:
    echo          %PY% scripts\publish_results.py --stage ^<N^>
)
if not "%LEFT_COMMITS%"=="0" (
    echo      %LEFT_COMMITS% commit^(s^) made here but not pushed:
    git log --oneline "@{upstream}..HEAD"
    echo      The same command pushes them, rebasing, and refuses if one touches code:
    echo          %PY% scripts\publish_results.py --stage ^<N^>
)
exit /b 1

:update_checkout
REM Puts an existing checkout on %BRANCH% and pulls. Returns 1 only when running on
REM would mean running code that is not on the remote at all.
git fetch --prune origin
if errorlevel 1 (
    echo   WARNING: could not reach the remote; continuing with the local copy.
    exit /b 0
)
REM per-edge-objective became main and was deleted on 2026-09-25. Its history IS main's,
REM so a checkout still standing on it moves across with its commits, pushed or not.
git rev-parse --verify --quiet refs/heads/per-edge-objective >NUL
if not errorlevel 1 (
    echo   Moving this checkout from per-edge-objective to %BRANCH% ^(one time^).
    git checkout -B %BRANCH% per-edge-objective || exit /b 1
    git branch --set-upstream-to=origin/%BRANCH% || exit /b 1
    git branch -d per-edge-objective
) else (
    git checkout %BRANCH% || exit /b 1
)
REM A local main from before the 2026-08-25 history compression shares no commit with
REM origin/main. The pull below would fail, and the old warning then ran that code anyway.
git merge-base HEAD origin/%BRANCH% >NUL
if errorlevel 1 (
    echo   ERROR: local %BRANCH% shares no history with origin/%BRANCH%; it predates
    echo   the 2026-08-25 history compression. Replace it, then run this again:
    echo       git checkout -B %BRANCH% origin/%BRANCH%
    exit /b 1
)
git pull --ff-only
if errorlevel 1 echo   WARNING: pull failed, continuing with the local copy.
exit /b 0

:fail
echo.
echo ============================================================
echo  STOPPED - the step above did not finish.
echo  Check the matching log in results\logs, and send it back.
echo ============================================================
echo.
pause
exit /b 1
