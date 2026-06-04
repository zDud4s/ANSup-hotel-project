"""ARCHIVED / NOT WIRED INTO run_all. Superseded by run_gaussian.py, which is the
canonical GMM (full covariance, fit-on-full, n_init=10) feeding experiments.csv and
the report. This file (diag covariance, 100k-subsample fit) is retained only for
provenance and is intentionally excluded from the pipeline. Do not cite its numbers.

Task 2 GMM clustering.

GMM likelihood uses component covariance, while all internal indices below
use hard labels from ``predict()`` and Euclidean distances on the same X
matrix used to fit the model. This keeps metric governance aligned with the
k-means/iK-means baseline for cross-family comparison.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from ..data.validate import load_raw
from ..evaluation.metrics import compute_indices, mean_pairwise_ari
from ..preprocessing.feature_config import FAST_MODE, FAST_N, FAST_SEED, SEEDS
from ..preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)
from ..utils.experiment_logger import EXP_HEADER, append_experiments
from .run_baseline import (
    SCALERS,
    SILHOUETTE_SAMPLE_FAST,
    SILHOUETTE_SAMPLE_FULL,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR   = PROJECT_ROOT / "tables"
FIGURES_DIR  = PROJECT_ROOT / "figures"
EXP_CSV      = PROJECT_ROOT / "experiments.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE = list(range(2, 9))
GMM_FULL_FIT_N = 100000
GMM_PARAMS = dict(
    covariance_type="diag",
    n_init=1,
    init_params="k-means++",
    reg_covar=1e-6,
    max_iter=100,
    tol=1e-3,
)


def _progress(message: str) -> None:
    print(message, flush=True)


def _row(**kwargs) -> dict:
    return {field: kwargs.get(field, "") for field in EXP_HEADER}


def run_gmm_collect(
    X: np.ndarray,
    variant: str,
    sil_sample: int | None,
    fit_sample_size: int | None = None,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    labels_cache: dict = {}
    fit_note = ""
    X_fit = X
    if fit_sample_size is not None and len(X) > fit_sample_size:
        sample_idx = np.random.default_rng(FAST_SEED).choice(
            len(X),
            size=fit_sample_size,
            replace=False,
        )
        X_fit = X[sample_idx]
        fit_note = f"fit_sample_n={fit_sample_size}; scored_n={len(X)}"
        _progress(f"    [GMM] fitting sample {X_fit.shape}; scoring full X {X.shape}")

    total_runs = len(K_RANGE) * len(SEEDS)
    run_no = 0
    for k in K_RANGE:
        _progress(f"    [GMM] starting k={k}")
        labels_by_seed: dict[int, np.ndarray] = {}
        per_seed: dict[int, dict] = {}

        for seed in SEEDS:
            run_no += 1
            _progress(
                f"    [GMM {run_no}/{total_runs}] fit "
                f"variant={variant} k={k} seed={seed}"
            )
            gmm = GaussianMixture(n_components=k, random_state=seed, **GMM_PARAMS)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                gmm.fit(X_fit)
            labels = gmm.predict(X)
            has_conv_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            _progress(
                f"    [GMM {run_no}/{total_runs}] metrics "
                f"variant={variant} k={k} seed={seed}"
            )
            idx = compute_indices(X, labels, silhouette_sample_size=sil_sample, seed=seed)

            labels_by_seed[seed] = labels
            per_seed[seed] = {
                **idx,
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
                "log_likelihood": float(gmm.score(X) * len(X)),
                "n_iter": int(gmm.n_iter_),
                "converged": bool(gmm.converged_) and not has_conv_warning,
            }
            metrics = per_seed[seed]
            _progress(
                f"    [GMM {run_no}/{total_runs}] done "
                f"sil={metrics['silhouette']:.4f} "
                f"bic={metrics['bic']:.2f} "
                f"aic={metrics['aic']:.2f} "
                f"converged={metrics['converged']}"
            )

        ref = labels_by_seed[SEEDS[0]]
        pairwise_ari = mean_pairwise_ari(labels_by_seed)
        _progress(f"    [GMM] aggregating ARI for k={k}")
        for seed in SEEDS:
            metrics = per_seed[seed]
            row = _row(
                task="2.0",
                method="GMM",
                variant=variant,
                k=k,
                seed=seed,
                silhouette=metrics["silhouette"],
                calinski_harabasz=metrics["calinski_harabasz"],
                davies_bouldin=metrics["davies_bouldin"],
                ari_vs_seed0=adjusted_rand_score(ref, labels_by_seed[seed]),
                bic=metrics["bic"],
                aic=metrics["aic"],
                log_likelihood=metrics["log_likelihood"],
                n_iter=metrics["n_iter"],
                converged=metrics["converged"],
                notes=fit_note,
            )
            row["_pairwise_ari"] = pairwise_ari
            rows.append(row)
        labels_cache[("GMM", variant, k)] = labels_by_seed
        _progress(f"    [GMM] completed k={k}")

    return rows, labels_cache


def build_gmm_summary(rows_dicts: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows_dicts)
    numeric_cols = [
        "k", "silhouette", "calinski_harabasz", "davies_bouldin",
        "bic", "aic", "log_likelihood", "n_iter", "ari_vs_seed0", "_pairwise_ari",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["converged"] = df["converged"].astype(str).str.lower().isin(["true", "1"])

    summary = (df.groupby(["method", "variant", "k"])
               .agg(sil_mean=("silhouette", "mean"),
                    sil_std=("silhouette", "std"),
                    ch_mean=("calinski_harabasz", "mean"),
                    ch_std=("calinski_harabasz", "std"),
                    db_mean=("davies_bouldin", "mean"),
                    db_std=("davies_bouldin", "std"),
                    bic_mean=("bic", "mean"),
                    bic_std=("bic", "std"),
                    aic_mean=("aic", "mean"),
                    aic_std=("aic", "std"),
                    loglik_mean=("log_likelihood", "mean"),
                    n_iter_mean=("n_iter", "mean"),
                    converged_frac=("converged", "mean"),
                    ari_mean=("_pairwise_ari", "mean"))
               .reset_index())
    return summary[[
        "method", "variant", "k", "sil_mean", "sil_std", "ch_mean", "ch_std",
        "db_mean", "db_std", "bic_mean", "bic_std", "aic_mean", "aic_std",
        "loglik_mean", "n_iter_mean", "converged_frac", "ari_mean",
    ]]


def select_k_star(summary_df: pd.DataFrame) -> list[dict]:
    selections: list[dict] = []
    for variant, sub in summary_df.groupby("variant"):
        sub = sub.sort_values("k").copy()
        bic_ranked = sub.sort_values("bic_mean")
        bic_best = bic_ranked.iloc[0]
        monotone_decreasing = bool((sub["bic_mean"].diff().dropna() < 0).all())

        chosen = bic_best
        gate_passed = bool(bic_best["ari_mean"] >= 0.80)
        justification = "BIC minimum passed ARI gate."
        if not gate_passed:
            passing = bic_ranked[bic_ranked["ari_mean"] >= 0.80]
            if not passing.empty:
                chosen = passing.iloc[0]
                justification = "BIC minimum failed ARI gate; selected next BIC-ranked k passing ARI >= 0.80."
            else:
                justification = "No k passed ARI >= 0.80; retained BIC minimum with stability caveat."

        if monotone_decreasing:
            justification += " BIC decreases through k=8; bounded-search caveat."

        tied = sub[sub["bic_mean"] <= bic_best["bic_mean"] + bic_best["bic_std"]]
        if len(tied) > 1:
            justification += " BIC candidates fall within one std; numeric rule retained."

        selections.append({
            "variant": variant,
            "k_argmin_bic": int(bic_best["k"]),
            "bic_at_argmin": float(bic_best["bic_mean"]),
            "ari_at_argmin": float(bic_best["ari_mean"]),
            "gate_passed": gate_passed,
            "k_star": int(chosen["k"]),
            "justification": justification,
        })
    return selections


def plot_aic_bic(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, mean_col, std_col, title in [
        (axes[0], "aic_mean", "aic_std", "AIC (lower = better)"),
        (axes[1], "bic_mean", "bic_std", "BIC (lower = better)"),
    ]:
        for variant, sub in summary.groupby("variant"):
            sub = sub.sort_values("k")
            ax.errorbar(sub["k"], sub[mean_col], yerr=sub[std_col],
                        marker="o", capsize=3, label=f"scaler={variant}")
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Task 2 - GMM AIC/BIC vs k (mean +/- std over 5 seeds)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_gmm_aic_bic.png", bbox_inches="tight", dpi=130)
    plt.close()


def plot_internal_indices(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    metrics = [
        ("sil_mean", "sil_std", "Silhouette (higher = better)"),
        ("ch_mean", "ch_std", "Calinski-Harabasz (higher = better)"),
        ("db_mean", "db_std", "Davies-Bouldin (lower = better)"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for variant, sub in summary.groupby("variant"):
            sub = sub.sort_values("k")
            ax.errorbar(sub["k"], sub[mean_col], yerr=sub[std_col],
                        marker="o", capsize=3, label=f"scaler={variant}")
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Task 2 - GMM internal indices vs k (mean +/- std over 5 seeds)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_gmm_internal_indices.png", bbox_inches="tight", dpi=130)
    plt.close()


def plot_stability(stab_rows: pd.DataFrame | list[dict]) -> None:
    stab = pd.DataFrame(stab_rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for variant, sub in stab.groupby("variant"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["ari_mean"], marker="o", label=f"scaler={variant}")
    ax.axhline(0.80, color="grey", ls="--", lw=1, label="ARI = 0.80 (selection rule)")
    ax.set_xlabel("k")
    ax.set_ylabel("mean pairwise ARI across seeds")
    ax.set_title("Task 2 - GMM stability vs k", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_gmm_stability.png", bbox_inches="tight", dpi=130)
    plt.close()


def generate_profiles(X_input_df: pd.DataFrame,
                      profiling_frame: pd.DataFrame,
                      labels: np.ndarray,
                      variant: str,
                      k_star: int) -> None:
    profile_source = X_input_df.copy()
    for col in profiling_frame.columns:
        if col not in profile_source.columns:
            profile_source[col] = profiling_frame[col]
    profile_source["cluster"] = labels

    for feature in ["hotel", "customer_type", "meal", "deposit_type", "market_segment"]:
        tab = pd.crosstab(profile_source["cluster"],
                          profile_source[feature].fillna("Unknown"),
                          normalize="index")
        tab = tab.sort_index().reset_index()
        tab.to_csv(TABLES_DIR / f"task2_gmm_profile_{variant}_{feature}.csv", index=False)

        plot_df = tab.set_index("cluster")
        ax = plot_df.plot(kind="bar", stacked=True, figsize=(8, 4))
        ax.set_xlabel("cluster")
        ax.set_ylabel("share within cluster")
        ax.set_title(f"GMM profile: {feature} (scaler={variant}, k={k_star})", fontsize=10)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"task2_gmm_profile_{variant}_{feature}.png",
                    bbox_inches="tight", dpi=130)
        plt.close()


def main(fast: bool = FAST_MODE) -> None:
    _progress("=== Task 2 - GMM clustering ===")
    sil_sample = SILHOUETTE_SAMPLE_FAST if fast else SILHOUETTE_SAMPLE_FULL
    fit_sample_size = None if fast else GMM_FULL_FIT_N
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        _progress(f"[FULL MODE] {len(df):,} rows")
    df = add_cyclic_seasonality(df)
    X_input, profiling_frame = split_clustering_and_profiling(df)

    all_rows: list[dict] = []
    all_labels: dict = {}

    for variant in SCALERS:
        _progress(f"\n--- variant: scaler={variant} ---")
        preproc = build_preprocessor(variant)
        _progress(f"  [GMM] fitting preprocessor for scaler={variant}")
        X = preproc.fit_transform(X_input)
        _progress(f"X shape: {X.shape}")
        _progress(f"  [GMM] k in {K_RANGE} x seeds {SEEDS}")
        if fit_sample_size is not None:
            _progress(f"  [GMM] full mode fit sample size: {fit_sample_size:,}")
        rows, labels = run_gmm_collect(X, variant, sil_sample, fit_sample_size)
        all_rows.extend(rows)
        all_labels.update(labels)

    _progress("\nAppending GMM rows to experiments.csv ...")
    append_experiments(all_rows, EXP_CSV)

    summary = build_gmm_summary(all_rows)
    summary.to_csv(TABLES_DIR / "task2_gmm_summary.csv", index=False)
    selections = select_k_star(summary)
    selection_df = pd.DataFrame(selections, columns=[
        "variant", "k_argmin_bic", "bic_at_argmin", "ari_at_argmin",
        "gate_passed", "k_star", "justification",
    ])
    selection_df.to_csv(TABLES_DIR / "task2_gmm_selection.csv", index=False)

    _progress("\nRendering GMM figures ...")
    plot_aic_bic(summary)
    plot_internal_indices(summary)
    plot_stability(summary[["variant", "k", "ari_mean"]])

    for _, selection in selection_df.iterrows():
        variant = selection["variant"]
        k_star = int(selection["k_star"])
        _progress(f"  [GMM profiles] variant={variant} k_star={k_star}")
        labels = all_labels[("GMM", variant, k_star)][SEEDS[0]]
        generate_profiles(X_input, profiling_frame, labels, variant, k_star)

    _progress("\nDone.")
    _progress(f"  tables/  : task2_gmm_summary.csv, task2_gmm_selection.csv")
    _progress(f"  figures/ : task2_gmm_aic_bic.png, task2_gmm_internal_indices.png, task2_gmm_stability.png")


if __name__ == "__main__":
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    else:
        fast = FAST_MODE
    main(fast=fast)
