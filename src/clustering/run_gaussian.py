"""Task 2.1 — Gaussian Mixture Model (GMM) with AIC/BIC Selection.

Runs a Gaussian Mixture Model (EM) over components k in {2..8} across the
predefined seeds and scaling variants (StandardScaler and RobustScaler).
Computes internal verification metrics alongside AIC, BIC, and Log-Likelihood
to facilitate distribution-based model selection.

Outputs:
- Updated experiments.csv with GMM runs
- task2_1_gmm_raw_summary.csv: Dedicated CSV with GMM run details
- task2_1_gmm_selection_curves.png: Plots of AIC and B
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..evaluation.metrics import compute_indices
from ..preprocessing.feature_config import FAST_MODE, SEEDS
from .run_baseline import RunRow, build_summary_tables, load_clustering_input, SCALERS
from ..preprocessing.pipeline import build_preprocessor
from ..preprocessing.feature_config import FAST_SEED
from ..utils.experiment_logger import append_experiments, build_run_meta, to_parameters_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR   = PROJECT_ROOT / "tables"
FIGURES_DIR  = PROJECT_ROOT / "figures"
EXP_CSV      = PROJECT_ROOT / "experiments.csv"

K_RANGE = list(range(2, 9))
SILHOUETTE_SAMPLE_FAST = 2_000
SILHOUETTE_SAMPLE_FULL = 5_000

def _progress(message: str) -> None:
    print(message, flush=True)

def run_gmm_collect(X: np.ndarray, variant: str, sil_sample: int | None) -> list[RunRow]:
    rows: list[RunRow] = []
    total_runs = len(K_RANGE) * len(SEEDS)
    run_no = 0
    gmm_params = to_parameters_json({
        "algorithm": "GaussianMixture",
        "covariance_type": "full", "max_iter": 200, "n_init": 10,
        "scaler": variant, "k_range": [K_RANGE[0], K_RANGE[-1]],
        "distance": "euclidean (hard-label indices)",
    })

    for k in K_RANGE:
        _progress(f"    [GMM] starting k={k}")
        for seed in SEEDS:
            run_no += 1
            _progress(f"    [GMM {run_no}/{total_runs}] fitting variant={variant} k={k} seed={seed}")
            
            gmm = GaussianMixture(
                n_components=k, 
                covariance_type="full", 
                random_state=seed, 
                max_iter=200, 
                n_init=10
            )
            
            labels = gmm.fit_predict(X)
            
            aic_val = gmm.aic(X)
            bic_val = gmm.bic(X)
            log_lik = gmm.score(X) * X.shape[0] 
            idx = compute_indices(X, labels, silhouette_sample_size=sil_sample, seed=seed)
            
            rows.append(RunRow(
                task="2.1", 
                method="GaussianMixture", 
                variant=variant,
                k=k, 
                seed=seed,
                silhouette=idx["silhouette"],
                calinski_harabasz=idx["calinski_harabasz"],
                davies_bouldin=idx["davies_bouldin"],
                ari_vs_seed0=0.0, 
                aic=f"{aic_val:.2f}",
                bic=f"{bic_val:.2f}",
                log_likelihood=f"{log_lik:.2f}",
                n_iter=str(gmm.n_iter_),
                converged=str(gmm.converged_),
                parameters=gmm_params,
                notes="Covariance: full"
            ))
    return rows

def plot_internal_indices(summary: pd.DataFrame) -> None:
    km = summary[summary["method"] == "GaussianMixture"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    metrics = [("sil_mean", "sil_std", "Silhouette (higher = better)"),
               ("ch_mean", "ch_std", "Calinski-Harabasz (higher = better)"),
               ("db_mean", "db_std", "Davies-Bouldin (lower = better)")]
    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for variant, sub in km.groupby("variant"):
            sub = sub.sort_values("k")
            ax.errorbar(sub["k"], sub[mean_col], yerr=sub[std_col],
                        marker="o", capsize=3, label=f"scaler={variant}")
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Task 2.1 - Gaussian Mixture Model internal indices vs k (mean +/- std over {len(SEEDS)} seeds)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_1_gmm_internal_indices.png",
                bbox_inches="tight", dpi=130)
    plt.close()

def plot_gmm_selection_curves(df_gmm: pd.DataFrame) -> None:
    """Plots AIC and BIC curves to reveal the optimal component knee-point."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    
    df_gmm = df_gmm.copy()
    df_gmm["aic"] = df_gmm["aic"].astype(float)
    df_gmm["bic"] = df_gmm["bic"].astype(float)
    
    summary = df_gmm.groupby(["variant", "k"])[["aic", "bic"]].mean().reset_index()
    
    for variant, sub in summary.groupby("variant"):
        sub = sub.sort_values("k")
        axes[0].plot(sub["k"], sub["bic"], marker="s", label=f"BIC (scaler={variant})")
        axes[1].plot(sub["k"], sub["aic"], marker="o", label=f"AIC (scaler={variant})")
        
    axes[0].set_title("BIC Score (Lower = Better / Parsimonious)")
    axes[0].set_xlabel("Number of Components (k)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    
    axes[1].set_title("AIC Score (Lower = Better / Predictive)")
    axes[1].set_xlabel("Number of Components (k)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    
    fig.suptitle("Task 2.1 - Gaussian Mixture Model Selection Curves", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_1_gmm_selection_curves.png", bbox_inches="tight", dpi=130)
    plt.close()

def main(fast: bool = FAST_MODE) -> None:
    _progress("=== Task 2.1 - GMM Clustering and Model Selection ===")
    sil_sample = SILHOUETTE_SAMPLE_FAST if fast else SILHOUETTE_SAMPLE_FULL
    df_input = load_clustering_input(fast)
    run_meta = build_run_meta(fast, n_rows=len(df_input), seed=FAST_SEED if fast else None)
    _progress(f"  run_id={run_meta['run_id']}  sample_rule={run_meta['sample_rule']}")

    gmm_rows: list[RunRow] = []
    
    for variant in SCALERS:
        _progress(f"\n--- variant: scaler={variant} ---")
        preproc = build_preprocessor(SCALERS[variant])
        X = preproc.fit_transform(df_input)
        
        rows = run_gmm_collect(X, variant, sil_sample)
        gmm_rows.extend(rows)
        
    _progress("\nAppending GMM runs to experiments.csv ...")
    append_experiments([r.as_dict() for r in gmm_rows], EXP_CSV, run_meta=run_meta)
    
    summary, _ = build_summary_tables(gmm_rows, stability=None)
    df_gmm = pd.DataFrame([r.as_dict() for r in gmm_rows])
    df_gmm.to_csv(TABLES_DIR / "task2_1_gmm_raw_summary.csv", index=False)
    
    _progress("Generating Information Criterion plots...")
    plot_gmm_selection_curves(df_gmm)
    _progress("Generating internal indices plots...")
    plot_internal_indices(summary)
    _progress("Done.")

if __name__ == "__main__":
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    else:
        fast = FAST_MODE
    main(fast=fast)