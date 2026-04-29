"""Task 1.2 — baseline clustering.

Runs MiniBatchKMeans for k in {2..8} with the configured fixed seeds, plus iK-means
under the same protocol, on two preprocessing variants (StandardScaler
and RobustScaler). For each (method, variant, k, seed) combination it
records Silhouette, Calinski-Harabasz, Davies-Bouldin in the same
metric space as clustering, plus mean pairwise ARI across seeds as the
stability measure.

Outputs:
    experiments.csv                                       (one row per fit)
    tables/task1_2_summary.csv                            (mean +/- std per k/method/variant)
    tables/task1_2_stability.csv                          (mean ARI per k/method/variant)
    figures/task1_2_kmeans_internal_indices.png           (Sil/CH/DB vs k, both scalers)
    figures/task1_2_kmeans_stability.png                  (ARI vs k, both scalers)
    figures/task1_2_ikmeans_summary.png                   (iK-means indices, both scalers)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..data.validate import load_raw
from ..evaluation.metrics import compute_indices, mean_pairwise_ari
from ..preprocessing.feature_config import (
    FAST_MODE,
    FAST_N,
    FAST_SEED,
    OHE_MIN_PREVALENCE,
    SEEDS,
)
from ..preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    get_feature_names,
    split_clustering_and_profiling,
)
from ..utils.experiment_logger import EXP_HEADER, write_experiments
from .ikmeans import fit_ikmeans

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR   = PROJECT_ROOT / "tables"
FIGURES_DIR  = PROJECT_ROOT / "figures"
EXP_CSV      = PROJECT_ROOT / "experiments.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# predefined parameter ranges
K_RANGE = list(range(2, 9))                # k in {2, 3, 4, 5, 6, 7, 8}
SCALERS = {"standard": StandardScaler, "robust": RobustScaler}

# Silhouette is O(n^2). We subsample so the pairwise-distance matrix stays
# tractable (a 10k x 10k float64 matrix is ~750 MB; many environments
# fragment well below that).
SILHOUETTE_SAMPLE_FAST = 2_000
SILHOUETTE_SAMPLE_FULL = 5_000


def _progress(message: str) -> None:
    print(message, flush=True)


@dataclass
class RunRow:
    task: str
    method: str
    variant: str
    k: int
    seed: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    ari_vs_seed0: float
    bic: str = ""
    aic: str = ""
    log_likelihood: str = ""
    n_iter: str = ""
    converged: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        return {field: getattr(self, field) for field in EXP_HEADER}


def load_clustering_input(fast: bool) -> pd.DataFrame:
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        _progress(f"[FULL MODE] {len(df):,} rows")
    df = add_cyclic_seasonality(df)
    X_input, _ = split_clustering_and_profiling(df)
    return X_input


def build_summary_tables(rows: list[RunRow], stability: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame([r.as_dict() for r in rows])
    summary = (df.groupby(["method", "variant", "k"])[["silhouette",
                                                       "calinski_harabasz",
                                                       "davies_bouldin"]]
               .agg(["mean", "std"])
               .round(4)
               .reset_index())
    summary.columns = ["method", "variant", "k",
                       "sil_mean", "sil_std",
                       "ch_mean", "ch_std",
                       "db_mean", "db_std"]
    stab_df = pd.DataFrame(stability)
    return summary, stab_df


def collect_stability(rows: list[RunRow], labels_cache: dict) -> list[dict]:
    """Mean pairwise ARI per (method, variant, k)."""
    out = []
    for (method, variant, k), labels_by_seed in labels_cache.items():
        out.append({
            "method": method,
            "variant": variant,
            "k": k,
            "mean_pairwise_ari": round(mean_pairwise_ari(labels_by_seed), 4),
        })
    return out


def run_kmeans_collect(X: np.ndarray, variant: str, sil_sample: int | None
                       ) -> tuple[list[RunRow], dict]:
    """Like run_kmeans but also returns labels_by_seed for stability analysis."""
    rows: list[RunRow] = []
    labels_cache: dict = {}
    from sklearn.metrics import adjusted_rand_score
    total_runs = len(K_RANGE) * len(SEEDS)
    run_no = 0
    for k in K_RANGE:
        _progress(f"    [k-means] starting k={k}")
        labels_by_seed: dict[int, np.ndarray] = {}
        per_seed_indices: dict[int, dict] = {}
        for seed in SEEDS:
            run_no += 1
            _progress(
                f"    [k-means {run_no}/{total_runs}] fit "
                f"variant={variant} k={k} seed={seed}"
            )
            mb = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                                 batch_size=1024, max_iter=300)
            labels = mb.fit_predict(X)
            labels_by_seed[seed] = labels
            _progress(
                f"    [k-means {run_no}/{total_runs}] metrics "
                f"variant={variant} k={k} seed={seed}"
            )
            per_seed_indices[seed] = compute_indices(X, labels,
                                                    silhouette_sample_size=sil_sample,
                                                    seed=seed)
            idx = per_seed_indices[seed]
            _progress(
                f"    [k-means {run_no}/{total_runs}] done "
                f"sil={idx['silhouette']:.4f} "
                f"ch={idx['calinski_harabasz']:.2f} "
                f"db={idx['davies_bouldin']:.4f}"
            )
        ref = labels_by_seed[SEEDS[0]]
        _progress(f"    [k-means] aggregating ARI for k={k}")
        for seed in SEEDS:
            ari_ref = adjusted_rand_score(ref, labels_by_seed[seed])
            idx = per_seed_indices[seed]
            rows.append(RunRow(
                task="1.2", method="MiniBatchKMeans", variant=variant,
                k=k, seed=seed,
                silhouette=idx["silhouette"],
                calinski_harabasz=idx["calinski_harabasz"],
                davies_bouldin=idx["davies_bouldin"],
                ari_vs_seed0=ari_ref,
            ))
        labels_cache[("MiniBatchKMeans", variant, k)] = labels_by_seed
        _progress(f"    [k-means] completed k={k}")
    return rows, labels_cache


def run_ikmeans_collect(X: np.ndarray, variant: str, sil_sample: int | None
                        ) -> tuple[list[RunRow], dict]:
    rows: list[RunRow] = []
    labels_cache: dict = {}
    labels_by_seed: dict[int, np.ndarray] = {}
    per_seed_k: dict[int, int] = {}
    per_seed_indices: dict[int, dict] = {}
    from sklearn.metrics import adjusted_rand_score
    for run_no, seed in enumerate(SEEDS, start=1):
        _progress(
            f"    [iK-means {run_no}/{len(SEEDS)}] fit "
            f"variant={variant} seed={seed}"
        )
        labels, _, k_auto = fit_ikmeans(X, seed=seed, k_max=K_RANGE[-1])
        labels_by_seed[seed] = labels
        per_seed_k[seed] = k_auto
        _progress(
            f"    [iK-means {run_no}/{len(SEEDS)}] metrics "
            f"variant={variant} seed={seed} k_auto={k_auto}"
        )
        per_seed_indices[seed] = compute_indices(X, labels,
                                                 silhouette_sample_size=sil_sample,
                                                 seed=seed)
        idx = per_seed_indices[seed]
        _progress(
            f"    [iK-means {run_no}/{len(SEEDS)}] done "
            f"sil={idx['silhouette']:.4f} "
            f"ch={idx['calinski_harabasz']:.2f} "
            f"db={idx['davies_bouldin']:.4f}"
        )
    ref = labels_by_seed[SEEDS[0]]
    _progress("    [iK-means] aggregating ARI")
    for seed in SEEDS:
        ari_ref = adjusted_rand_score(ref, labels_by_seed[seed])
        idx = per_seed_indices[seed]
        rows.append(RunRow(
            task="1.2", method="iKMeans", variant=variant,
            k=per_seed_k[seed], seed=seed,
            silhouette=idx["silhouette"],
            calinski_harabasz=idx["calinski_harabasz"],
            davies_bouldin=idx["davies_bouldin"],
            ari_vs_seed0=ari_ref,
            notes="auto-determined k",
        ))
    # Group iK-means by the modal k for stability reporting; if seeds disagree,
    # ARI is still meaningful (different partitions on the same X).
    modal_k = max(set(per_seed_k.values()), key=lambda v: list(per_seed_k.values()).count(v))
    labels_cache[("iKMeans", variant, modal_k)] = labels_by_seed
    return rows, labels_cache


def plot_internal_indices(summary: pd.DataFrame) -> None:
    km = summary[summary["method"] == "MiniBatchKMeans"]
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
    fig.suptitle(f"Task 1.2 - k-means internal indices vs k (mean +/- std over {len(SEEDS)} seeds)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_2_kmeans_internal_indices.png",
                bbox_inches="tight", dpi=130)
    plt.close()


def plot_stability(stab_df: pd.DataFrame) -> None:
    km = stab_df[stab_df["method"] == "MiniBatchKMeans"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for variant, sub in km.groupby("variant"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["mean_pairwise_ari"], marker="o", label=f"scaler={variant}")
    ax.axhline(0.80, color="grey", ls="--", lw=1, label="ARI = 0.80 (selection rule)")
    ax.set_xlabel("k")
    ax.set_ylabel("mean pairwise ARI across seeds")
    ax.set_title("Task 1.2 - k-means stability vs k", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_2_kmeans_stability.png",
                bbox_inches="tight", dpi=130)
    plt.close()


def plot_ikmeans(summary: pd.DataFrame, stab_df: pd.DataFrame) -> None:
    ik = summary[summary["method"] == "iKMeans"].copy()
    ik_stab = stab_df[stab_df["method"] == "iKMeans"].copy()
    if ik.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar([f"{v}\nk={int(k)}" for v, k in zip(ik["variant"], ik["k"])],
                ik["sil_mean"], yerr=ik["sil_std"], capsize=4, color="steelblue")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title(f"iK-means Silhouette (mean +/- std, {len(SEEDS)} seeds)", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar([f"{v}\nk={int(k)}" for v, k in zip(ik_stab["variant"], ik_stab["k"])],
                ik_stab["mean_pairwise_ari"], color="seagreen")
    axes[1].axhline(0.80, color="grey", ls="--", lw=1)
    axes[1].set_ylabel("mean pairwise ARI")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("iK-means stability (mean ARI across seeds)", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_2_ikmeans_summary.png",
                bbox_inches="tight", dpi=130)
    plt.close()


def main(fast: bool = FAST_MODE) -> None:
    _progress("=== Task 1.2 - baseline clustering ===")
    sil_sample = SILHOUETTE_SAMPLE_FAST if fast else SILHOUETTE_SAMPLE_FULL
    df_input = load_clustering_input(fast)

    all_rows: list[RunRow] = []
    all_labels: dict = {}

    for variant in SCALERS:
        _progress(f"\n--- variant: scaler={variant} ---")
        preproc = build_preprocessor(SCALERS[variant])
        _progress(f"  [preprocess] fitting transformer for scaler={variant}")
        X = preproc.fit_transform(df_input)
        feature_names = get_feature_names(preproc)
        block_weight = preproc.named_steps["block"].weight_
        n_num = len(feature_names) - preproc.named_steps["block"].n_cat_
        _progress(
            f"X shape: {X.shape} "
            f"(num={n_num}, cat_kept={preproc.named_steps['block'].n_cat_}, "
            f"block_weight={block_weight:.3f}, "
            f"OHE prevalence floor={OHE_MIN_PREVALENCE})"
        )

        _progress(f"  [k-means] k in {K_RANGE} x seeds {SEEDS}")
        km_rows, km_labels = run_kmeans_collect(X, variant, sil_sample)
        all_rows.extend(km_rows)
        all_labels.update(km_labels)

        _progress(f"  [iK-means] seeds {SEEDS}, k_max={K_RANGE[-1]}")
        ik_rows, ik_labels = run_ikmeans_collect(X, variant, sil_sample)
        all_rows.extend(ik_rows)
        all_labels.update(ik_labels)

    _progress("\nWriting experiments.csv ...")
    write_experiments([r.as_dict() for r in all_rows], EXP_CSV)
    stability = collect_stability(all_rows, all_labels)
    summary, stab_df = build_summary_tables(all_rows, stability)
    summary.to_csv(TABLES_DIR / "task1_2_summary.csv", index=False)
    stab_df.to_csv(TABLES_DIR / "task1_2_stability.csv", index=False)

    _progress("\nRendering figures ...")
    plot_internal_indices(summary)
    plot_stability(stab_df)
    plot_ikmeans(summary, stab_df)

    _progress("\nDone.")
    _progress(f"  experiments.csv : {EXP_CSV}")
    _progress(f"  tables/         : task1_2_summary.csv, task1_2_stability.csv")
    _progress(f"  figures/        : task1_2_kmeans_internal_indices.png, "
              f"task1_2_kmeans_stability.png, task1_2_ikmeans_summary.png")


if __name__ == "__main__":
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    else:
        fast = FAST_MODE
    main(fast=fast)
