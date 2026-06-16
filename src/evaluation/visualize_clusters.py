"""2D cluster-space visualisation.

Projects the fitted clustering space to two principal components and colours
points by cluster assignment. The plot is diagnostic only: PCA preserves the
largest linear variance directions, not all distances in the original space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..data.validate import load_raw
from ..preprocessing.feature_config import FAST_MODE, FAST_N, FAST_SEED
from ..preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)
from ..clustering.ikmeans import fit_ikmeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_inputs(fast: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        _progress(f"[FULL MODE] {len(df):,} rows")
    df = add_cyclic_seasonality(df)
    return split_clustering_and_profiling(df)


def _sample_indices(n: int, sample_size: int) -> np.ndarray:
    if sample_size >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(FAST_SEED).choice(n, size=sample_size, replace=False))


def _cluster_palette(n_clusters: int) -> list[str]:
    cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n_clusters)]


def plot_ikmeans_robust_pca(fast: bool, sample_size: int, seed: int) -> None:
    _progress("=== Cluster-space PCA visualisation ===")
    x_input, profiling_frame = _load_inputs(fast)

    _progress("Fitting RobustScaler preprocessing")
    preproc = build_preprocessor("robust")
    x = preproc.fit_transform(x_input)
    _progress(f"X shape: {x.shape}")

    _progress(f"Fitting iKMeans seed={seed}")
    labels, centres, k_auto = fit_ikmeans(x, seed=seed)
    counts = pd.Series(labels).value_counts().sort_index()
    _progress(f"iKMeans k={k_auto}; cluster sizes: {counts.to_dict()}")

    sample_idx = _sample_indices(len(x), sample_size)
    x_sample = x[sample_idx]
    labels_sample = labels[sample_idx]

    _progress(f"Fitting PCA on plotted sample n={len(sample_idx):,}")
    pca = PCA(n_components=2, random_state=FAST_SEED)
    coords = pca.fit_transform(x_sample)
    centre_coords = pca.transform(centres)
    explained = pca.explained_variance_ratio_

    profile_cols = ["hotel", "customer_type", "market_segment", "deposit_type"]
    export = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "cluster": labels_sample,
    })
    for col in profile_cols:
        if col in x_input.columns:
            export[col] = x_input.iloc[sample_idx][col].to_numpy()
        elif col in profiling_frame.columns:
            export[col] = profiling_frame.iloc[sample_idx][col].to_numpy()
    export_path = TABLES_DIR / "task2_cluster_space_pca_ikmeans_robust_sample.csv"
    export.to_csv(export_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    palette = _cluster_palette(k_auto)
    for ax, title in zip(axes, ["Full projection", "Central 98% zoom"]):
        for cluster_id in range(k_auto):
            mask = labels_sample == cluster_id
            share = counts.get(cluster_id, 0) / len(labels)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=7,
                alpha=0.28,
                linewidths=0,
                color=palette[cluster_id],
                label=f"cluster {cluster_id} ({share:.1%})",
            )
        ax.scatter(
            centre_coords[:, 0],
            centre_coords[:, 1],
            s=160,
            marker="X",
            color="black",
            edgecolor="white",
            linewidth=1.0,
            label="cluster centres",
            zorder=5,
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)

    x_low, x_high = np.quantile(coords[:, 0], [0.01, 0.99])
    y_low, y_high = np.quantile(coords[:, 1], [0.01, 0.99])
    pad_x = (x_high - x_low) * 0.08
    pad_y = (y_high - y_low) * 0.08
    axes[1].set_xlim(x_low - pad_x, x_high + pad_x)
    axes[1].set_ylim(y_low - pad_y, y_high + pad_y)
    handles, labels_for_legend = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels_for_legend, fontsize=8, loc="best", frameon=True)
    fig.suptitle("iKMeans + RobustScaler clusters projected with PCA", fontsize=12, y=1.01)
    fig.text(
        0.01,
        0.01,
        "Diagnostic projection: PCA shows two linear directions; actual clustering used the full preprocessed feature space.",
        fontsize=8,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.035, 1, 1))

    fig_path = FIGURES_DIR / "task2_cluster_space_pca_ikmeans_robust.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    _progress(f"Saved figure: {fig_path}")
    _progress(f"Saved sample coordinates: {export_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fast = FAST_MODE
    if args.full:
        fast = False
    elif args.fast:
        fast = True

    plot_ikmeans_robust_pca(fast=fast, sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
