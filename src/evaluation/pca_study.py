"""Extension Module E4 - PCA/SVD representation study for clustering.

Compares the canonical StandardScaler clustering matrix against a PCA-reduced
representation under the same evaluation and stability protocol used by the
baseline clustering run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ..clustering.ikmeans import fit_ikmeans
from ..clustering.run_baseline import (
    K_RANGE,
    RunRow,
    SILHOUETTE_SAMPLE_FAST,
    SILHOUETTE_SAMPLE_FULL,
    load_clustering_input,
)
from ..evaluation.metrics import compute_indices, mean_pairwise_ari
from ..preprocessing.feature_config import (
    FAST_MODE,
    FAST_SEED,
    OHE_MIN_PREVALENCE,
    SEEDS,
)
from ..preprocessing.pipeline import build_preprocessor, get_feature_names
from ..utils.experiment_logger import append_experiments, build_run_meta, to_parameters_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXP_CSV = PROJECT_ROOT / "experiments.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {"orig": "standard_orig", "pca": "standard_pca"}
THRESHOLDS = (0.80, 0.90, 0.95)
N_KEEP_THRESHOLD = 0.90


def _progress(message: str) -> None:
    print(message, flush=True)


def build_standard_matrix(fast: bool) -> np.ndarray:
    """Build the same StandardScaler clustering matrix used by the baseline."""
    df_input = load_clustering_input(fast)
    preproc = build_preprocessor(StandardScaler)
    _progress("  [preprocess] fitting baseline StandardScaler transformer")
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
    return X


def fit_pca_projection(X: np.ndarray) -> tuple[PCA, np.ndarray, pd.DataFrame, pd.DataFrame, int]:
    _progress("  [PCA] fitting full SVD PCA on StandardScaler matrix")
    pca = PCA(svd_solver="full", random_state=SEEDS[0])
    X_all = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    dims = {
        threshold: int(np.searchsorted(cumulative, threshold, side="left") + 1)
        for threshold in THRESHOLDS
    }
    n_keep = dims[N_KEEP_THRESHOLD]

    explained = pd.DataFrame({
        "component": np.arange(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative": cumulative,
    })
    selection = pd.DataFrame({
        "threshold": list(dims.keys()),
        "n_components": list(dims.values()),
    })

    _progress("  [PCA] explained-variance threshold dimensions:")
    for threshold in THRESHOLDS:
        _progress(f"    {threshold:.2f}: {dims[threshold]} components")
    _progress(f"  [PCA] chosen n_keep={n_keep} for cumulative variance >= {N_KEEP_THRESHOLD:.2f}")
    return pca, X_all[:, :n_keep], explained, selection, n_keep


def run_kmeans_space(X: np.ndarray, space: str, sil_sample: int | None) -> tuple[list[RunRow], list[dict]]:
    variant = VARIANTS[space]
    rows: list[RunRow] = []
    summary_rows: list[dict] = []
    total_runs = len(K_RANGE) * len(SEEDS)
    run_no = 0
    params = to_parameters_json({
        "algorithm": "MiniBatchKMeans",
        "n_init": 10,
        "batch_size": 1024,
        "max_iter": 300,
        "scaler": "standard",
        "space": space,
        "k_range": [K_RANGE[0], K_RANGE[-1]],
        "distance": "euclidean",
    })

    for k in K_RANGE:
        labels_by_seed: dict[int, np.ndarray] = {}
        per_seed_indices: dict[int, dict] = {}
        _progress(f"    [{space} k-means] starting k={k}")
        for seed in SEEDS:
            run_no += 1
            _progress(f"    [{space} k-means {run_no}/{total_runs}] fit k={k} seed={seed}")
            mb = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
                batch_size=1024,
                max_iter=300,
            )
            labels = mb.fit_predict(X)
            labels_by_seed[seed] = labels
            _progress(f"    [{space} k-means {run_no}/{total_runs}] metrics k={k} seed={seed}")
            per_seed_indices[seed] = compute_indices(
                X,
                labels,
                silhouette_sample_size=sil_sample,
                seed=seed,
            )

        ref = labels_by_seed[SEEDS[0]]
        for seed in SEEDS:
            idx = per_seed_indices[seed]
            rows.append(RunRow(
                task="E4",
                method="MiniBatchKMeans",
                variant=variant,
                k=k,
                seed=seed,
                silhouette=idx["silhouette"],
                calinski_harabasz=idx["calinski_harabasz"],
                davies_bouldin=idx["davies_bouldin"],
                ari_vs_seed0=adjusted_rand_score(ref, labels_by_seed[seed]),
                parameters=params,
            ))

        sils = [per_seed_indices[seed]["silhouette"] for seed in SEEDS]
        chs = [per_seed_indices[seed]["calinski_harabasz"] for seed in SEEDS]
        dbs = [per_seed_indices[seed]["davies_bouldin"] for seed in SEEDS]
        summary_rows.append({
            "space": space,
            "method": "MiniBatchKMeans",
            "variant": variant,
            "k": k,
            "sil_mean": float(np.nanmean(sils)),
            "sil_std": float(np.nanstd(sils, ddof=1)),
            "ch_mean": float(np.nanmean(chs)),
            "db_mean": float(np.nanmean(dbs)),
            "ari_mean": float(mean_pairwise_ari(labels_by_seed)),
        })
        _progress(f"    [{space} k-means] completed k={k}")
    return rows, summary_rows


def run_ikmeans_space(X: np.ndarray, space: str, sil_sample: int | None) -> tuple[list[RunRow], list[dict]]:
    variant = VARIANTS[space]
    rows: list[RunRow] = []
    labels_by_seed: dict[int, np.ndarray] = {}
    per_seed_k: dict[int, int] = {}
    per_seed_indices: dict[int, dict] = {}
    params = to_parameters_json({
        "algorithm": "iKMeans",
        "k_max": K_RANGE[-1],
        "min_cluster_size": "max(20, 0.5% of n)",
        "final_kmeans_max_iter": 300,
        "scaler": "standard",
        "space": space,
        "distance": "euclidean",
        "init": "anomalous-pattern (deterministic)",
    })

    for run_no, seed in enumerate(SEEDS, start=1):
        _progress(f"    [{space} iK-means {run_no}/{len(SEEDS)}] fit seed={seed}")
        labels, _, k_auto = fit_ikmeans(X, seed=seed, k_max=K_RANGE[-1])
        labels_by_seed[seed] = labels
        per_seed_k[seed] = k_auto
        _progress(f"    [{space} iK-means {run_no}/{len(SEEDS)}] metrics seed={seed} k_auto={k_auto}")
        per_seed_indices[seed] = compute_indices(
            X,
            labels,
            silhouette_sample_size=sil_sample,
            seed=seed,
        )

    ref = labels_by_seed[SEEDS[0]]
    for seed in SEEDS:
        idx = per_seed_indices[seed]
        rows.append(RunRow(
            task="E4",
            method="iKMeans",
            variant=variant,
            k=per_seed_k[seed],
            seed=seed,
            silhouette=idx["silhouette"],
            calinski_harabasz=idx["calinski_harabasz"],
            davies_bouldin=idx["davies_bouldin"],
            ari_vs_seed0=adjusted_rand_score(ref, labels_by_seed[seed]),
            parameters=params,
            notes="auto-determined k",
        ))

    modal_k = max(set(per_seed_k.values()), key=lambda v: list(per_seed_k.values()).count(v))
    sils = [per_seed_indices[seed]["silhouette"] for seed in SEEDS]
    chs = [per_seed_indices[seed]["calinski_harabasz"] for seed in SEEDS]
    dbs = [per_seed_indices[seed]["davies_bouldin"] for seed in SEEDS]
    summary = [{
        "space": space,
        "method": "iKMeans",
        "variant": variant,
        "k": modal_k,
        "sil_mean": float(np.nanmean(sils)),
        "sil_std": float(np.nanstd(sils, ddof=1)),
        "ch_mean": float(np.nanmean(chs)),
        "db_mean": float(np.nanmean(dbs)),
        "ari_mean": float(mean_pairwise_ari(labels_by_seed)),
    }]
    _progress(f"    [{space} iK-means] completed modal_k={modal_k}")
    return rows, summary


def write_tables(explained: pd.DataFrame, selection: pd.DataFrame, comparison: pd.DataFrame) -> None:
    outputs = [
        (explained, TABLES_DIR / "e4_pca_explained_variance.csv"),
        (selection, TABLES_DIR / "e4_pca_dimension_selection.csv"),
        (comparison, TABLES_DIR / "e4_pca_vs_original.csv"),
    ]
    for df, path in outputs:
        df.to_csv(path, index=False)
        _progress(f"Saved {path}")


def plot_scree(explained: pd.DataFrame, n_keep: int) -> None:
    max_components = min(30, len(explained))
    sub = explained.iloc[:max_components]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(
        sub["component"],
        sub["explained_variance_ratio"],
        color="#4C78A8",
        alpha=0.8,
        label="per-component variance",
    )
    ax1.set_xlabel("PCA component")
    ax1.set_ylabel("explained variance ratio")
    ax1.grid(alpha=0.25, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(
        sub["component"],
        sub["cumulative"],
        color="#F58518",
        marker="o",
        linewidth=1.8,
        label="cumulative variance",
    )
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_ylim(0, 1.02)
    for threshold in THRESHOLDS:
        ax2.axhline(threshold, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
        ax2.text(max_components + 0.25, threshold, f"{threshold:.0%}", va="center", fontsize=8)
    if n_keep <= max_components:
        ax1.axvline(n_keep, color="#54A24B", linestyle="-", linewidth=1.5, label=f"n_keep={n_keep}")
        ax1.annotate(
            f"n_keep={n_keep}",
            xy=(n_keep, sub.loc[sub["component"].eq(n_keep), "explained_variance_ratio"].iloc[0]),
            xytext=(n_keep + 1, ax1.get_ylim()[1] * 0.8),
            arrowprops={"arrowstyle": "->", "color": "#54A24B"},
            fontsize=9,
            color="#2F6B2F",
        )
    else:
        ax1.text(
            0.98,
            0.92,
            f"n_keep={n_keep} (beyond first {max_components})",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#2F6B2F",
        )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax1.set_title("E4 PCA scree and cumulative variance threshold selection")
    plt.tight_layout()
    path = FIGURES_DIR / "e4_pca_scree.png"
    plt.savefig(path, bbox_inches="tight", dpi=140)
    plt.close()
    _progress(f"Saved {path}")


def plot_comparison(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    colors = {"orig": "#4C78A8", "pca": "#F58518"}
    labels = {"orig": "original StandardScaler", "pca": "PCA reduced"}

    for space in ("orig", "pca"):
        km = comparison[
            (comparison["space"] == space)
            & (comparison["method"] == "MiniBatchKMeans")
        ].sort_values("k")
        axes[0].plot(km["k"], km["sil_mean"], marker="o", color=colors[space], label=labels[space])
        axes[1].plot(km["k"], km["ari_mean"], marker="o", color=colors[space], label=labels[space])

        ik = comparison[
            (comparison["space"] == space)
            & (comparison["method"] == "iKMeans")
        ]
        axes[0].scatter(
            ik["k"],
            ik["sil_mean"],
            marker="D",
            s=52,
            color=colors[space],
            edgecolor="black",
            linewidth=0.6,
            label=f"{labels[space]} iK-means",
        )
        axes[1].scatter(
            ik["k"],
            ik["ari_mean"],
            marker="D",
            s=52,
            color=colors[space],
            edgecolor="black",
            linewidth=0.6,
        )

    axes[0].set_ylabel("Silhouette mean")
    axes[0].set_title("A. Cluster quality: MiniBatchKMeans by k with iK-means points")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Mean pairwise ARI")
    axes[1].set_title("B. Stability across seeds")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")

    fig.suptitle("E4 PCA vs original clustering representation", y=0.995, fontsize=12)
    plt.tight_layout()
    path = FIGURES_DIR / "e4_pca_vs_original.png"
    plt.savefig(path, bbox_inches="tight", dpi=140)
    plt.close()
    _progress(f"Saved {path}")


def run(fast: bool = FAST_MODE) -> None:
    _progress("=== E4 - PCA/SVD clustering study ===")
    sil_sample = SILHOUETTE_SAMPLE_FAST if fast else SILHOUETTE_SAMPLE_FULL
    X = build_standard_matrix(fast)
    run_meta = build_run_meta(fast, n_rows=X.shape[0], seed=FAST_SEED if fast else None)
    _progress(f"  run_id={run_meta['run_id']}  sample_rule={run_meta['sample_rule']}")

    _, X_pca, explained, selection, n_keep = fit_pca_projection(X)
    spaces = {"orig": X, "pca": X_pca}

    all_rows: list[RunRow] = []
    comparison_rows: list[dict] = []
    for space, X_space in spaces.items():
        _progress(f"\n--- space: {space} ({VARIANTS[space]}) ---")
        km_rows, km_summary = run_kmeans_space(X_space, space, sil_sample)
        all_rows.extend(km_rows)
        comparison_rows.extend(km_summary)
        ik_rows, ik_summary = run_ikmeans_space(X_space, space, sil_sample)
        all_rows.extend(ik_rows)
        comparison_rows.extend(ik_summary)

    _progress("\nAppending E4 fits to experiments.csv ...")
    append_experiments([row.as_dict() for row in all_rows], EXP_CSV, run_meta=run_meta)

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison[
        ["space", "method", "variant", "k", "sil_mean", "sil_std", "ch_mean", "db_mean", "ari_mean"]
    ].sort_values(["method", "space", "k"])

    write_tables(explained, selection, comparison)
    plot_scree(explained, n_keep)
    plot_comparison(comparison)

    _progress("\nSummary:")
    _progress(f"  chosen n_keep={n_keep} at cumulative explained variance >= {N_KEEP_THRESHOLD:.2f}")
    _progress(
        "  Caveat: variance preservation does not guarantee preservation of "
        "cluster-separating directions; the empirical comparison below is what decides whether PCA helps."
    )
    _progress("Done.")


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast)


if __name__ == "__main__":
    main()
