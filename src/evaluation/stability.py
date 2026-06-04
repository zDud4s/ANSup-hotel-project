"""Bootstrap/subsample stability for the headline clustering methods.

This module contrasts seed-only ARI with data-perturbation stability. The
bootstrap protocol uses subsampling without replacement so the score reflects
whether the reference partition structure is reproduced after removing 20% of
bookings and refitting the model.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..clustering.ikmeans import fit_ikmeans
from ..clustering.run_baseline import load_clustering_input
from ..evaluation.metrics import mean_pairwise_ari
from ..preprocessing.feature_config import FAST_MODE, FAST_SEED, OHE_MIN_PREVALENCE, SEEDS
from ..preprocessing.pipeline import build_preprocessor, get_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KMEANS_K = 7
GMM_K = 8
IKMEANS_K_MAX = 8
SUBSAMPLE_FRACTION = 0.8
REPRESENTATION = "StandardScaler"
STABILITY_GATE = 0.80


def _progress(message: str) -> None:
    print(message, flush=True)


def build_standard_matrix(fast: bool) -> np.ndarray:
    """Build the same StandardScaler clustering matrix used by pca_study.py."""
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


def fit_kmeans(X: np.ndarray, seed: int) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=KMEANS_K,
        random_state=seed,
        n_init=10,
        batch_size=1024,
        max_iter=300,
    )
    return model.fit_predict(X)


def fit_gmm(X: np.ndarray, seed: int, n_init: int) -> np.ndarray:
    model = GaussianMixture(
        n_components=GMM_K,
        covariance_type="full",
        random_state=seed,
        max_iter=200,
        n_init=n_init,
    )
    return model.fit_predict(X)


def fit_method(X: np.ndarray, method: str, seed: int, *, gmm_n_init: int) -> tuple[np.ndarray, int]:
    if method == "kmeans":
        return fit_kmeans(X, seed), KMEANS_K
    if method == "ikmeans":
        labels, _, k_auto = fit_ikmeans(X, seed=seed, k_max=IKMEANS_K_MAX)
        return labels, int(k_auto)
    if method == "gmm":
        return fit_gmm(X, seed, n_init=gmm_n_init), GMM_K
    raise ValueError(f"Unknown method: {method}")


def pairwise_ari_std(labels_by_seed: dict[int, np.ndarray]) -> float:
    seeds = sorted(labels_by_seed.keys())
    scores = [
        adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])
        for a, b in combinations(seeds, 2)
    ]
    if len(scores) < 2:
        return float("nan")
    return float(np.std(scores, ddof=1))


def reference_partitions(X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    _progress("\n--- Step A: reference partitions on the full matrix ---")
    ref_labels: dict[str, np.ndarray] = {}
    ref_k: dict[str, int] = {}
    for method in ("kmeans", "ikmeans", "gmm"):
        _progress(f"  [reference] fitting {method} seed={SEEDS[0]}")
        labels, k = fit_method(X, method, SEEDS[0], gmm_n_init=10)
        ref_labels[method] = labels
        ref_k[method] = k
        _progress(f"  [reference] {method}: k={k}")
    return ref_labels, ref_k


def bootstrap_stability(X: np.ndarray, ref_labels: dict[str, np.ndarray]) -> dict[str, list[float]]:
    _progress("\n--- Step B: subsample stability ---")
    n = X.shape[0]
    n_sub = int(round(SUBSAMPLE_FRACTION * n))
    scores = {method: [] for method in ("kmeans", "ikmeans", "gmm")}

    for round_no, seed in enumerate(SEEDS, start=1):
        idx = np.random.default_rng(seed).choice(n, size=n_sub, replace=False)
        _progress(
            f"  [bootstrap {round_no}/{len(SEEDS)}] "
            f"seed={seed} subsample={n_sub:,}/{n:,}"
        )
        for method in ("kmeans", "ikmeans", "gmm"):
            _progress(f"    [{method}] refit on subsample")
            boot_labels, k = fit_method(X[idx], method, seed, gmm_n_init=1)
            ari = adjusted_rand_score(ref_labels[method][idx], boot_labels)
            scores[method].append(float(ari))
            _progress(f"    [{method}] k={k} bootstrap_ari={ari:.4f}")

    return scores


def seed_stability(X: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    _progress("\n--- Step C: seed-based stability on the full matrix ---")
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for method in ("kmeans", "gmm"):
        labels_by_seed: dict[int, np.ndarray] = {}
        for run_no, seed in enumerate(SEEDS, start=1):
            _progress(f"  [{method} seed {run_no}/{len(SEEDS)}] fitting seed={seed}")
            labels, _ = fit_method(X, method, seed, gmm_n_init=10)
            labels_by_seed[seed] = labels
        means[method] = float(mean_pairwise_ari(labels_by_seed))
        stds[method] = pairwise_ari_std(labels_by_seed)
        _progress(
            f"  [{method}] seed_ari_mean={means[method]:.4f} "
            f"seed_ari_std={stds[method]:.4f}"
        )

    means["ikmeans"] = 1.0
    stds["ikmeans"] = 0.0
    _progress("  [ikmeans] seed_ari_mean=1.0000 seed_ari_std=0.0000 (structural)")
    return means, stds


def build_table(
    boot_scores: dict[str, list[float]],
    seed_means: dict[str, float],
    seed_stds: dict[str, float],
    ref_k: dict[str, int],
) -> pd.DataFrame:
    notes = {
        "kmeans": "headline k=7",
        "ikmeans": "deterministic; seed-ARI is structural, not an empirical stability signal",
        "gmm": "reference fit uses n_init=10; bootstrap fits use n_init=1 for tractability",
    }
    rows = []
    for method in ("kmeans", "ikmeans", "gmm"):
        vals = np.asarray(boot_scores[method], dtype=float)
        rows.append({
            "method": method,
            "representation": REPRESENTATION,
            "k": ref_k[method],
            "n_bootstrap": len(SEEDS),
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "ari_boot_mean": float(np.mean(vals)),
            "ari_boot_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "ari_seed_mean": seed_means[method],
            "ari_seed_std": seed_stds[method],
            "note": notes[method],
        })
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame) -> Path:
    path = TABLES_DIR / "stability_bootstrap.csv"
    df.to_csv(path, index=False)
    _progress(f"Saved {path}")
    return path


def plot_stability(df: pd.DataFrame) -> Path:
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        df["ari_seed_mean"],
        width,
        yerr=df["ari_seed_std"],
        capsize=4,
        color="#4C78A8",
        label="Seed ARI",
    )
    ax.bar(
        x + width / 2,
        df["ari_boot_mean"],
        width,
        yerr=df["ari_boot_std"],
        capsize=4,
        color="#F58518",
        label="Bootstrap/subsample ARI",
    )
    ax.axhline(
        STABILITY_GATE,
        color="#666666",
        linestyle="--",
        linewidth=1.1,
        label="stability gate",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Seed vs data-perturbation stability "
        f"(subsample fraction={SUBSAMPLE_FRACTION:.1f}, B={len(SEEDS)})"
    )
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Note: iK-means seed-ARI is structural; bootstrap ARI is the empirical stability signal.",
        fontsize=8,
        color="#444444",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIGURES_DIR / "stability_bootstrap.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    _progress(f"Saved {path}")
    return path


def run(fast: bool = FAST_MODE) -> None:
    _progress("=== Bootstrap (data-perturbation) stability ===")
    _progress(f"  mode={'fast' if fast else 'full'} seed_for_fast_sample={FAST_SEED if fast else 'n/a'}")
    X = build_standard_matrix(fast)
    ref_labels, ref_k = reference_partitions(X)
    boot_scores = bootstrap_stability(X, ref_labels)
    seed_means, seed_stds = seed_stability(X)
    df = build_table(boot_scores, seed_means, seed_stds, ref_k)

    _progress("\nSummary:")
    for row in df.itertuples(index=False):
        _progress(
            f"  {row.method}: "
            f"bootstrap ARI={row.ari_boot_mean:.4f} +/- {row.ari_boot_std:.4f}; "
            f"seed ARI={row.ari_seed_mean:.4f} +/- {row.ari_seed_std:.4f}"
        )

    write_table(df)
    plot_stability(df)
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
