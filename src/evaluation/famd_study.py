"""RQ2 FAMD representation comparison for governed mixed-data clustering.

Compares three representations of the same governed mixed-data inputs under a
single protocol:

* one-hot + block weight, the current baseline representation;
* PCA retaining 90% variance on that baseline representation;
* FAMD retaining 90% variance, with categorical variables encoded using the
  MCA chi-square metric.

The FAMD categorical coding centers each dummy column by its prevalence and
scales by ``1 / sqrt(prevalence)``. This matches the FactoMineR/prince FAMD
convention: quantitative variables contribute roughly one unit of inertia,
while a categorical variable with K retained levels contributes roughly K - 1.
High-cardinality variables therefore carry more inertia; the existing
governance mitigates this through rare grouping/capping before one-hot
encoding, with country already capped or held outside the clustering inputs.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

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
    SILHOUETTE_SAMPLE_FAST,
    SILHOUETTE_SAMPLE_FULL,
    load_clustering_input,
)
from ..evaluation.metrics import compute_indices
from ..preprocessing.feature_config import (
    CLUSTER_NUMERICAL,
    FAST_MODE,
    FAST_SEED,
    OHE_MIN_PREVALENCE,
    SEEDS,
)
from ..preprocessing.pipeline import build_preprocessor, get_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (0.80, 0.90, 0.95)
N_KEEP_THRESHOLD = 0.90
SUBSAMPLE_FRACTION = 0.8
STABILITY_GATE = 0.80
REPRESENTATIONS = ("onehot_blockweight", "pca90", "famd90")
METHODS = ("kmeans_k7", "ikmeans")


def _progress(message: str) -> None:
    print(message, flush=True)


def _escalate(reason: str, needed: str, partial_output: str = "") -> None:
    print("## Escalation", flush=True)
    print(f"reason: {reason}", flush=True)
    print(f"needed: {needed}", flush=True)
    print("suggested-next: inspect the preprocessing or evaluation contract before rerunning.", flush=True)
    if partial_output:
        print("partial-output:", flush=True)
        print(partial_output, flush=True)
    raise SystemExit(1)


def verify_compute_indices_contract() -> None:
    sig = inspect.signature(compute_indices)
    params = list(sig.parameters.values())
    expected = ["X", "labels", "silhouette_sample_size", "seed"]
    actual = [p.name for p in params]
    if actual != expected:
        _escalate(
            "compute_indices signature differs from the expected FAMD protocol.",
            "Expected compute_indices(X, labels, silhouette_sample_size=..., seed=...).",
            f"actual signature: {sig}",
        )
    if (
        params[2].default is inspect._empty
        or params[3].default is inspect._empty
    ):
        _escalate(
            "compute_indices does not expose defaults for silhouette_sample_size and seed.",
            "Expected optional silhouette_sample_size and seed parameters.",
            f"actual signature: {sig}",
        )


def _threshold_dims(cumulative: np.ndarray, *, label: str) -> dict[float, int]:
    dims: dict[float, int] = {}
    for threshold in THRESHOLDS:
        if cumulative.size == 0 or cumulative[-1] + 1e-12 < threshold:
            _escalate(
                f"{label} PCA failed to reach {threshold:.0%} cumulative explained variance.",
                "Investigate the matrix rank/explained-variance vector.",
                f"cumulative variance: {np.array2string(cumulative, precision=6)}",
            )
        dims[threshold] = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    return dims


def build_representations(fast: bool) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[float, int], np.ndarray, np.ndarray]:
    """Build one-hot+block-weight, PCA90, and FAMD90 on the same governed rows."""
    df_input = load_clustering_input(fast)
    preproc = build_preprocessor(StandardScaler)
    _progress("  [preprocess] fitting governed StandardScaler transformer")
    preproc.fit(df_input)

    feature_names = get_feature_names(preproc)
    n_num = len(CLUSTER_NUMERICAL)
    actual_prefix = feature_names[:n_num]
    transformer_order = [(name, cols) for name, _, cols in preproc.named_steps["ct"].transformers_]
    if actual_prefix != CLUSTER_NUMERICAL:
        _escalate(
            "ColumnTransformer column ordering is not [numerics then categoricals].",
            "FAMD construction requires standardized numerics in columns 0..n_num-1.",
            f"actual transformer order: {transformer_order}\nfirst feature names: {actual_prefix}",
        )

    ct = preproc.named_steps["ct"]
    Xct = np.asarray(ct.transform(df_input), dtype=float)
    X_onehot = np.asarray(preproc.transform(df_input), dtype=float)
    Z_num = Xct[:, :n_num]
    G = Xct[:, n_num:]
    p = G.mean(axis=0)
    if np.any(p <= 0):
        bad = np.where(p <= 0)[0].tolist()
        _escalate(
            "FAMD encountered zero-prevalence dummy columns.",
            "Variance-floor governance should remove empty dummies before FAMD coding.",
            f"bad dummy column offsets: {bad}",
        )

    G_famd = (G - p) / np.sqrt(p)
    X_famd = np.hstack([Z_num, G_famd])
    block_weight = preproc.named_steps["block"].weight_
    _progress(
        f"X onehot_blockweight shape: {X_onehot.shape} "
        f"(num={n_num}, cat_kept={G.shape[1]}, block_weight={block_weight:.3f}, "
        f"OHE prevalence floor={OHE_MIN_PREVALENCE})"
    )
    _progress(f"X FAMD-coded shape: {X_famd.shape}")

    _progress("  [PCA] fitting pca90 on onehot_blockweight")
    pca = PCA(svd_solver="full", random_state=SEEDS[0])
    X_pca_all = pca.fit_transform(X_onehot)
    pca_cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_dims = _threshold_dims(pca_cumulative, label="onehot_blockweight")
    X_pca90 = X_pca_all[:, :pca_dims[N_KEEP_THRESHOLD]]
    _progress(f"  [PCA] pca90 n_keep={pca_dims[N_KEEP_THRESHOLD]}")

    _progress("  [FAMD] fitting full SVD PCA on FAMD-coded matrix")
    famd_pca = PCA(svd_solver="full", random_state=SEEDS[0])
    X_famd_all = famd_pca.fit_transform(X_famd)
    famd_evr = famd_pca.explained_variance_ratio_
    famd_cumulative = np.cumsum(famd_evr)
    famd_dims = _threshold_dims(famd_cumulative, label="FAMD")
    famd_n_keep = famd_dims[N_KEEP_THRESHOLD]
    X_famd90 = X_famd_all[:, :famd_n_keep]
    _progress(f"  [FAMD] n_keep@90%={famd_n_keep}")

    inertia_rows = [{
        "component": "note",
        "explained_variance_ratio": np.nan,
        "cumulative": np.nan,
        "note": (
            f"n_keep@0.80={famd_dims[0.80]}; "
            f"n_keep@0.90={famd_dims[0.90]}; "
            f"n_keep@0.95={famd_dims[0.95]}"
        ),
    }]
    inertia_rows.extend({
        "component": int(i),
        "explained_variance_ratio": float(evr),
        "cumulative": float(cum),
        "note": "",
    } for i, (evr, cum) in enumerate(zip(famd_evr, famd_cumulative), start=1))
    inertia = pd.DataFrame(inertia_rows)

    reps = {
        "onehot_blockweight": X_onehot,
        "pca90": X_pca90,
        "famd90": X_famd90,
    }
    return reps, inertia, famd_dims, X_famd_all, famd_evr


def fit_method(X: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, int]:
    if method == "kmeans_k7":
        model = MiniBatchKMeans(
            n_clusters=7,
            random_state=seed,
            n_init=10,
            batch_size=1024,
            max_iter=300,
        )
        return model.fit_predict(X), 7
    if method == "ikmeans":
        labels, _, k_auto = fit_ikmeans(X, seed=seed, k_max=8)
        return labels, int(k_auto)
    raise ValueError(f"Unknown method: {method}")


def bootstrap_ari(X: np.ndarray, method: str, reference: np.ndarray) -> tuple[float, float, list[float]]:
    n = X.shape[0]
    n_sub = int(round(SUBSAMPLE_FRACTION * n))
    scores: list[float] = []
    for round_no, seed in enumerate(SEEDS, start=1):
        idx = np.random.default_rng(seed).choice(n, size=n_sub, replace=False)
        _progress(
            f"    [{method} bootstrap {round_no}/{len(SEEDS)}] "
            f"subsample={n_sub:,}/{n:,}"
        )
        boot_labels, k = fit_method(X[idx], method, seed)
        ari = adjusted_rand_score(reference[idx], boot_labels)
        scores.append(float(ari))
        _progress(f"    [{method} bootstrap] k={k} ari={ari:.4f}")
    arr = np.asarray(scores, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
    return float(np.mean(arr)), std, scores


def compare_representations(
    reps: dict[str, np.ndarray],
    sil_sample: int | None,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    rows: list[dict] = []
    baseline_ikmeans_labels: np.ndarray | None = None
    baseline_ikmeans_k: int | None = None

    for representation in REPRESENTATIONS:
        X = reps[representation]
        _progress(f"\n--- representation: {representation} ({X.shape[1]} dims) ---")
        for method in METHODS:
            _progress(f"  [{representation} {method}] fitting reference seed={SEEDS[0]}")
            labels, k = fit_method(X, method, SEEDS[0])
            if representation == "onehot_blockweight" and method == "ikmeans":
                baseline_ikmeans_labels = labels
                baseline_ikmeans_k = k

            _progress(f"  [{representation} {method}] computing indices")
            idx = compute_indices(
                X,
                labels,
                silhouette_sample_size=sil_sample,
                seed=SEEDS[0],
            )
            ari_mean, ari_std, _ = bootstrap_ari(X, method, labels)
            rows.append({
                "representation": representation,
                "method": method,
                "n_dims": int(X.shape[1]),
                "silhouette": idx["silhouette"],
                "calinski_harabasz": idx["calinski_harabasz"],
                "davies_bouldin": idx["davies_bouldin"],
                "ari_bootstrap_mean": ari_mean,
                "ari_bootstrap_std": ari_std,
                "note": f"reference k={k}; bootstrap B={len(SEEDS)}, frac={SUBSAMPLE_FRACTION:.1f}",
            })
            _progress(
                f"  [{representation} {method}] k={k} "
                f"sil={idx['silhouette']:.4f} boot_ari={ari_mean:.4f}"
            )

    if baseline_ikmeans_labels is None or baseline_ikmeans_k is None:
        raise RuntimeError("Internal error: baseline iK-means labels were not produced.")
    comparison = pd.DataFrame(rows)
    comparison = comparison[
        [
            "representation",
            "method",
            "n_dims",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "ari_bootstrap_mean",
            "ari_bootstrap_std",
            "note",
        ]
    ].sort_values(["method", "representation"])
    return comparison, baseline_ikmeans_labels, baseline_ikmeans_k


def write_tables(inertia: pd.DataFrame, comparison: pd.DataFrame) -> None:
    outputs = [
        (inertia, TABLES_DIR / "famd_inertia.csv"),
        (comparison, TABLES_DIR / "famd_representation_comparison.csv"),
    ]
    for df, path in outputs:
        df.to_csv(path, index=False)
        _progress(f"Saved {path}")


def plot_comparison(comparison: pd.DataFrame) -> None:
    colors = {
        "onehot_blockweight": "#4C78A8",
        "pca90": "#F58518",
        "famd90": "#54A24B",
    }
    labels = {
        "onehot_blockweight": "One-hot + block weight",
        "pca90": "PCA90 on one-hot",
        "famd90": "FAMD90",
    }
    method_labels = {"kmeans_k7": "MiniBatchKMeans k=7", "ikmeans": "iK-means"}
    methods = list(METHODS)
    x = np.arange(len(methods))
    width = 0.24

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for offset_no, rep in enumerate(REPRESENTATIONS):
        sub = comparison.set_index(["method", "representation"])
        positions = x + (offset_no - 1) * width
        sil = [sub.loc[(method, rep), "silhouette"] for method in methods]
        ari = [sub.loc[(method, rep), "ari_bootstrap_mean"] for method in methods]
        ari_err = [sub.loc[(method, rep), "ari_bootstrap_std"] for method in methods]

        axes[0].bar(positions, sil, width, color=colors[rep], label=labels[rep])
        axes[1].bar(
            positions,
            ari,
            width,
            yerr=ari_err,
            capsize=4,
            color=colors[rep],
            label=labels[rep],
        )

    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("A. Separation in each representation space")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8, loc="best")

    axes[1].axhline(
        STABILITY_GATE,
        color="#666666",
        linestyle="--",
        linewidth=1.1,
        label="0.80 gate",
    )
    axes[1].set_ylabel("Bootstrap ARI")
    axes[1].set_title("B. Subsample stability")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([method_labels[m] for m in methods])

    fig.suptitle("FAMD representation comparison for RQ2", y=0.995, fontsize=12)
    plt.tight_layout()
    path = FIGURES_DIR / "famd_representation_comparison.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    _progress(f"Saved {path}")


def plot_projection(
    X_famd_all: np.ndarray,
    famd_evr: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> None:
    n = X_famd_all.shape[0]
    n_plot = min(5_000, n)
    rng = np.random.default_rng(FAST_SEED)
    idx = rng.choice(n, size=n_plot, replace=False) if n_plot < n else np.arange(n)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    cmap = plt.get_cmap("tab10")
    for label in np.unique(labels[idx]):
        mask = labels[idx] == label
        ax.scatter(
            X_famd_all[idx][mask, 0],
            X_famd_all[idx][mask, 1],
            s=11,
            alpha=0.62,
            color=cmap(int(label) % 10),
            label=f"Segment {int(label)}",
            linewidths=0,
        )
    ax.set_xlabel(f"FAMD component 1 ({famd_evr[0] * 100:.1f}% inertia)")
    ax.set_ylabel(f"FAMD component 2 ({famd_evr[1] * 100:.1f}% inertia)")
    ax.set_title(
        f"Headline iK-means(k={k}) baseline segments projected into FAMD space"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2, frameon=True)
    plt.tight_layout()
    path = FIGURES_DIR / "famd_projection.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    _progress(f"Saved {path}")


def run(fast: bool = FAST_MODE) -> None:
    _progress("=== RQ2 - FAMD representation comparison ===")
    verify_compute_indices_contract()
    sil_sample = SILHOUETTE_SAMPLE_FAST if fast else SILHOUETTE_SAMPLE_FULL
    reps, inertia, famd_dims, X_famd_all, famd_evr = build_representations(fast)
    comparison, baseline_ikmeans_labels, baseline_ikmeans_k = compare_representations(
        reps,
        sil_sample,
    )
    write_tables(inertia, comparison)
    plot_comparison(comparison)
    plot_projection(X_famd_all, famd_evr, baseline_ikmeans_labels, baseline_ikmeans_k)

    _progress("\nFAMD threshold dimensions:")
    for threshold in THRESHOLDS:
        _progress(f"  n_keep@{threshold:.0%}={famd_dims[threshold]}")
    _progress("\nComparison table:")
    _progress(comparison.to_string(index=False))
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
