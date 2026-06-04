<#
run_all.ps1 — PowerShell equivalent of the bash run_all.

Reproduces all figures and tables end-to-end on Windows without going
through WSL. Use when bash run_all picks up WSL bash and cannot see the
Windows-side .venv.

Usage:
    .\run_all.ps1            # uses FAST_MODE from src/preprocessing/feature_config.py
    .\run_all.ps1 -Fast      # fast-check (5 000-row subsample, 10 seeds)
    .\run_all.ps1 -Full      # full mode (all rows, 10 seeds)
#>

[CmdletBinding()]
param(
    [switch]$Fast,
    [switch]$Full
)

$ErrorActionPreference = "Stop"

# --- Mode flag ----------------------------------------------------------------
$ModeFlag = $null
if ($Fast -and $Full) {
    Write-Error "Pass either -Fast or -Full, not both."
    exit 1
}
if ($Fast) { $ModeFlag = "--fast"; Write-Host "[run_all] Fast mode (5 000-row subsample)" }
elseif ($Full) { $ModeFlag = "--full"; Write-Host "[run_all] Full mode (all rows)" }
else { Write-Host "[run_all] No flag passed; using FAST_MODE from feature_config.py" }

# --- Run from the project root so 'src' is a resolvable package ---------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

# --- Resolve Python -----------------------------------------------------------
function Resolve-Python {
    if ($env:VIRTUAL_ENV) {
        $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    foreach ($candidate in @(
        "..\.venv\Scripts\python.exe",
        ".venv\Scripts\python.exe"
    )) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($cmd3) { return $cmd3.Source }
    return $null
}

$PythonBin = Resolve-Python
if (-not $PythonBin) {
    Write-Error "[run_all] Python interpreter not found. Activate the project venv or install Python."
    exit 1
}
Write-Host "[run_all] Using Python: $PythonBin"

# --- Avoid Windows 11 / Python 3.13 WMI hang in joblib/loky --------------------
if (-not $env:LOKY_MAX_CPU_COUNT) { $env:LOKY_MAX_CPU_COUNT = "8" }

# --- Pipeline steps -----------------------------------------------------------
function Assert-StepSucceeded {
    param([string]$Title)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[run_all] Step failed: $Title (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "=== Step 1: Data validation ==="
& $PythonBin -u -m src.data.validate
Assert-StepSucceeded "Step 1: Data validation"

Write-Host ""
Write-Host "=== Step 1b: EDA figure generation ==="
& $PythonBin -u -m src.data.eda $ModeFlag
Assert-StepSucceeded "Step 1b: EDA figure generation"

Write-Host ""
Write-Host "=== Step 2: Preprocessing ==="
& $PythonBin -u -m src.preprocessing.pipeline $ModeFlag
Assert-StepSucceeded "Step 2: Preprocessing"

Write-Host ""
Write-Host "=== Step 3: Task 1.2 baseline clustering ==="
& $PythonBin -u -m src.clustering.run_baseline $ModeFlag
Assert-StepSucceeded "Step 3: Task 1.2 baseline clustering"

Write-Host ""
Write-Host "=== Step 4: Task 2.1 Gaussian Mixture Model clustering ==="
& $PythonBin -u -m src.clustering.run_gaussian $ModeFlag
Assert-StepSucceeded "Step 4: Task 2.1 Gaussian Mixture Model clustering"

Write-Host ""
Write-Host "=== Step 4b: Bootstrap (data-perturbation) stability ==="
& $PythonBin -u -m src.evaluation.stability $ModeFlag
Assert-StepSucceeded "Step 4b: Bootstrap (data-perturbation) stability"

Write-Host ""
Write-Host "=== Step 4c: GMM extended-k BIC/AIC trajectory diagnostic ==="
& $PythonBin -u -m src.evaluation.gmm_model_selection $ModeFlag
Assert-StepSucceeded "Step 4c: GMM extended-k BIC/AIC trajectory diagnostic"

Write-Host ""
Write-Host "=== Step 5: iK-means RobustScaler cluster profile (anomaly-split check) ==="
& $PythonBin -u -m src.evaluation.profile_ikmeans $ModeFlag
Assert-StepSucceeded "Step 5: iK-means RobustScaler cluster profile (anomaly-split check)"

Write-Host ""
Write-Host "=== Step 6: StandardScaler vs RobustScaler profile comparison ==="
& $PythonBin -u -m src.evaluation.compare_scalers $ModeFlag
Assert-StepSucceeded "Step 6: StandardScaler vs RobustScaler profile comparison"

Write-Host ""
Write-Host "=== Step 7: Cross-family comparison ==="
& $PythonBin -u -m src.evaluation.compare_families $ModeFlag
Assert-StepSucceeded "Step 7: Cross-family comparison"

Write-Host ""
Write-Host "=== Step 8: Main-population clustering ==="
& $PythonBin -u -m src.evaluation.main_population_clustering $ModeFlag
Assert-StepSucceeded "Step 8: Main-population clustering"

Write-Host ""
Write-Host "=== Step 9: Cluster-space PCA visualisation ==="
& $PythonBin -u -m src.evaluation.visualize_clusters $ModeFlag
Assert-StepSucceeded "Step 9: Cluster-space PCA visualisation"

Write-Host ""
Write-Host "=== Step 9b: Cluster-space PCA(3) static + interactive visualiser ==="
& $PythonBin -u -m src.evaluation.visualize_components_3d $ModeFlag
Assert-StepSucceeded "Step 9b: Cluster-space PCA(3) static + interactive visualiser"

Write-Host ""
Write-Host "=== Step 10: E4 PCA/SVD clustering study ==="
& $PythonBin -u -m src.evaluation.pca_study $ModeFlag
Assert-StepSucceeded "Step 10: E4 PCA/SVD clustering study"

Write-Host ""
Write-Host "=== Step 10b: FAMD representation comparison (RQ2) ==="
& $PythonBin -u -m src.evaluation.famd_study $ModeFlag
Assert-StepSucceeded "Step 10b: FAMD representation comparison (RQ2)"

Write-Host ""
Write-Host "=== Step 10c: ADR-inclusion sensitivity (RQ2 governance) ==="
& $PythonBin -u -m src.evaluation.adr_sensitivity $ModeFlag
Assert-StepSucceeded "Step 10c: ADR-inclusion sensitivity (RQ2 governance)"

Write-Host ""
Write-Host "=== Step 10d: Gower + hierarchical (distinct non-Euclidean similarity, RQ2/RQ5) ==="
& $PythonBin -u -m src.evaluation.gower_hierarchical
Assert-StepSucceeded "Step 10d: Gower + hierarchical (distinct non-Euclidean similarity, RQ2/RQ5)"

Write-Host ""
Write-Host "=== Step 11: E1 cluster-aware anomaly analysis ==="
& $PythonBin -u -m src.evaluation.anomaly_analysis $ModeFlag
Assert-StepSucceeded "Step 11: E1 cluster-aware anomaly analysis"

Write-Host ""
Write-Host "=== Step 12: E5 t-SNE visualisation (viz-only) ==="
& $PythonBin -u -m src.evaluation.tsne_viz $ModeFlag
Assert-StepSucceeded "Step 12: E5 t-SNE visualisation (viz-only)"

Write-Host ""
Write-Host "=== Step 13: Headline iK-means+Standard k profiles ==="
& $PythonBin -u -m src.evaluation.profile_headline $ModeFlag
Assert-StepSucceeded "Step 13: Headline iK-means+Standard k profiles"

Write-Host ""
Write-Host "=== Step 14: Segment predictive utility (post-hoc cancellation) ==="
& $PythonBin -u -m src.evaluation.segment_predictive_utility $ModeFlag
Assert-StepSucceeded "Step 14: Segment predictive utility (post-hoc cancellation)"

Write-Host ""
Write-Host "=== Step 15: Surrogate decision-tree segment explanation ==="
& $PythonBin -u -m src.evaluation.surrogate_tree $ModeFlag
Assert-StepSucceeded "Step 15: Surrogate decision-tree segment explanation"

Write-Host ""
Write-Host "=== Done ==="
