"""I7 - Gower-distance hierarchical clustering robustness check.

This standalone diagnostic tests whether the headline 7-segment structure
replicates under a conceptually distinct mixed-data distance. Gower distance is
computed directly over governed numeric and categorical inputs: numeric
features are range-normalised internally, categorical features contribute
match/mismatch distances, and all features receive equal weight.

Average-linkage agglomerative clustering is the primary method because it can
operate on an arbitrary precomputed dissimilarity matrix. Ward linkage is
deliberately not used: Ward's variance objective assumes Euclidean geometry.
A complete-linkage fit is also reported as a robustness note.

Gower is O(n^2), so the module always uses the project's established fixed
FAST_N/FAST_SEED subsample rule before feature engineering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from ..clustering.ikmeans import fit_ikmeans
    from ..data.validate import load_raw
    from ..preprocessing.feature_config import (
        CLUSTER_CATEGORICAL,
        CLUSTER_NUMERICAL,
        COUNTRY_MIN_FREQ,
        FAST_N,
        FAST_SEED,
        SEEDS,
    )
    from ..preprocessing.pipeline import (
        RareCategoryGrouper,
        add_booking_features,
        add_cyclic_seasonality,
        build_preprocessor,
    )
except ImportError as exc:
    print("## Escalation", flush=True)
    print("reason: Required project import is unavailable.", flush=True)
    print(
        "needed: fit_ikmeans, build_preprocessor, RareCategoryGrouper, "
        "add_booking_features, add_cyclic_seasonality, load_raw, and feature_config constants.",
        flush=True,
    )
    print(f"actual-error: {exc}", flush=True)
    print(f"actual-name: {getattr(exc, 'name', '')}", flush=True)
    raise SystemExit(1) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_K = 7
SUMMARY_PATH = TABLES_DIR / "gower_hierarchical.csv"
DENDROGRAM_PATH = FIGURES_DIR / "gower_dendrogram.png"
CROSSTAB_PATH = FIGURES_DIR / "gower_vs_ikmeans_crosstab.png"


def _progress(message: str) -> None:
    print(message, flush=True)


def _escalate(reason: str, needed: str, partial_output: str = "") -> None:
    print("## Escalation", flush=True)
    print(f"reason: {reason}", flush=True)
    print(f"needed: {needed}", flush=True)
    if partial_output:
        print("partial-output:", flush=True)
        print(partial_output, flush=True)
    raise SystemExit(1)


def _load_fixed_subsample() -> pd.DataFrame:
    """Load raw bookings and apply the fixed FAST_N/FAST_SEED subsample rule."""
    df_raw = load_raw()
    df_raw = df_raw.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
    _progress(f"[FIXED SUBSAMPLE] {len(df_raw):,} rows (FAST_N={FAST_N}, FAST_SEED={FAST_SEED})")
    df = add_cyclic_seasonality(df_raw)
    df = add_booking_features(df)
    return df


def _prepare_gower_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute numerics and Unknown-fill/rare-group categoricals."""
    missing = [c for c in (CLUSTER_NUMERICAL + CLUSTER_CATEGORICAL) if c not in df.columns]
    if missing:
        _escalate(
            "Governed clustering columns are missing after feature engineering.",
            "Expected all CLUSTER_NUMERICAL and CLUSTER_CATEGORICAL columns to be present.",
            f"missing: {missing}",
        )

    numeric = df[CLUSTER_NUMERICAL].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())

    categorical = df[CLUSTER_CATEGORICAL].copy()
    categorical = categorical.fillna("Unknown").astype(str)
    grouper = RareCategoryGrouper(per_column={"country": COUNTRY_MIN_FREQ})
    grouped = grouper.fit_transform(categorical)
    categorical = pd.DataFrame(grouped, columns=CLUSTER_CATEGORICAL, index=df.index).astype(str)

    return numeric, categorical


def _build_gower_distance(numeric: pd.DataFrame, categorical: pd.DataFrame) -> tuple[np.ndarray, int, list[str]]:
    """Return a symmetric n x n Gower distance matrix in [0, 1]."""
    n_rows = len(numeric)
    if n_rows != len(categorical):
        _escalate(
            "Numeric and categorical Gower inputs have different row counts.",
            "Expected both governed input frames to describe the same fixed subsample.",
            f"numeric={len(numeric)}, categorical={len(categorical)}",
        )

    try:
        distance = np.zeros((n_rows, n_rows), dtype=np.float32)
        used_features = 0
        skipped_numeric: list[str] = []

        for col in CLUSTER_NUMERICAL:
            values = numeric[col].to_numpy(dtype=np.float32)
            value_range = float(np.max(values) - np.min(values))
            if not np.isfinite(value_range) or value_range == 0.0:
                skipped_numeric.append(col)
                continue
            distance += np.abs(values[:, None] - values[None, :]) / np.float32(value_range)
            used_features += 1

        for col in CLUSTER_CATEGORICAL:
            values = categorical[col].to_numpy()
            distance += (values[:, None] != values[None, :]).astype(np.float32)
            used_features += 1

        if used_features == 0:
            _escalate(
                "No usable features remained for Gower distance.",
                "Expected at least one non-constant numeric or categorical governed feature.",
                f"skipped_numeric={skipped_numeric}",
            )

        distance /= np.float32(used_features)
        np.fill_diagonal(distance, 0.0)
    except MemoryError as exc:
        _escalate(
            "Memory exhausted while building the Gower matrix.",
            "Gower is O(n^2); confirm the fixed subsample remains n=5000 or reduce only under explicit approval.",
            f"n={n_rows}, expected_n={FAST_N}, error={exc}",
        )

    _validate_gower(distance)
    return distance, used_features, skipped_numeric


def _validate_gower(distance: np.ndarray) -> None:
    diag_abs_max = float(np.max(np.abs(np.diag(distance))))
    min_value = float(np.min(distance))
    max_value = float(np.max(distance))
    symmetric = bool(np.allclose(distance, distance.T, atol=1e-6))
    zero_diag = diag_abs_max <= 1e-7
    in_unit_interval = min_value >= -1e-7 and max_value <= 1.0 + 1e-7

    if not (symmetric and zero_diag and in_unit_interval):
        _escalate(
            "Gower matrix validation failed.",
            "Expected D to be symmetric, zero-diagonal, and bounded in [0, 1].",
            (
                f"shape={distance.shape}, symmetric={symmetric}, "
                f"diag_abs_max={diag_abs_max:.8g}, min={min_value:.8g}, max={max_value:.8g}"
            ),
        )


def _fit_reference_partitions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit headline iK-means and k-means partitions on the exact same rows."""
    x_input = df[CLUSTER_NUMERICAL + CLUSTER_CATEGORICAL]
    preprocessor = build_preprocessor(StandardScaler)
    _progress("Fitting one-hot+block-weight StandardScaler reference matrix on the same rows")
    x_matrix = preprocessor.fit_transform(x_input)
    labels_ik, _, k_auto = fit_ikmeans(x_matrix, seed=SEEDS[0], k_max=8)
    labels_km = MiniBatchKMeans(
        n_clusters=TARGET_K,
        random_state=SEEDS[0],
        n_init=10,
        batch_size=1024,
        max_iter=300,
    ).fit_predict(x_matrix)
    _progress(f"Reference partitions: iK-means k_auto={k_auto}; MiniBatchKMeans k={TARGET_K}")
    return labels_ik, labels_km


def _cut_height_for_k(linkage_matrix: np.ndarray, k: int) -> float:
    if linkage_matrix.shape[0] < k - 1:
        return float(linkage_matrix[-1, 2])
    return float(linkage_matrix[-(k - 1), 2])


def _summarise_linkage(
    linkage_matrix: np.ndarray,
    distance: np.ndarray,
    labels_ik: np.ndarray,
    labels_km: np.ndarray,
    linkage_name: str,
    note: str,
) -> tuple[dict, np.ndarray]:
    labels = fcluster(linkage_matrix, t=TARGET_K, criterion="maxclust")
    sil = float(silhouette_score(distance, labels, metric="precomputed"))
    ari_ik = float(adjusted_rand_score(labels, labels_ik))
    ari_km = float(adjusted_rand_score(labels, labels_km))
    row = {
        "n_subsample": int(distance.shape[0]),
        "linkage": linkage_name,
        "k": int(np.unique(labels).size),
        "gower_silhouette": sil,
        "ari_vs_ikmeans": ari_ik,
        "ari_vs_kmeans": ari_km,
        "note": note,
    }
    return row, labels


def _plot_dendrogram(linkage_matrix: np.ndarray) -> None:
    cut_height = _cut_height_for_k(linkage_matrix, TARGET_K)
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        color_threshold=cut_height,
        above_threshold_color="#667085",
        ax=ax,
    )
    ax.axhline(cut_height, color="#C0392B", linestyle="--", linewidth=1.2, label=f"k={TARGET_K} cut")
    ax.set_title("Gower + average-linkage hierarchical (n=5000 fixed subsample)")
    ax.set_xlabel("Truncated cluster leaves")
    ax.set_ylabel("Average-linkage distance")
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(DENDROGRAM_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _progress(f"Saved {DENDROGRAM_PATH}")


def _plot_crosstab(labels_gower: np.ndarray, labels_ik: np.ndarray, ari: float) -> None:
    counts = pd.crosstab(
        pd.Series(labels_gower, name="Gower hierarchical"),
        pd.Series(labels_ik, name="iK-means headline"),
    ).sort_index(axis=0).sort_index(axis=1)
    row_share = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    fig_width = max(8.0, 1.0 + 0.8 * counts.shape[1])
    fig_height = max(5.5, 1.0 + 0.65 * counts.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(row_share.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row share")

    ax.set_xticks(np.arange(counts.shape[1]))
    ax.set_xticklabels([str(c) for c in counts.columns])
    ax.set_yticks(np.arange(counts.shape[0]))
    ax.set_yticklabels([str(r) for r in counts.index])
    ax.set_xlabel("iK-means headline segment")
    ax.set_ylabel("Gower-hierarchical cluster")
    ax.set_title(f"Gower hierarchical vs iK-means headline (counts annotated, ARI={ari:.3f})")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            share = float(row_share.iat[i, j])
            count = int(counts.iat[i, j])
            text_color = "white" if share >= 0.55 else "#1F2933"
            ax.text(j, i, f"{count:,}", ha="center", va="center", fontsize=8, color=text_color)

    fig.tight_layout()
    plt.savefig(CROSSTAB_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _progress(f"Saved {CROSSTAB_PATH}")


def _verdict_phrase(ari: float) -> str:
    if ari >= 0.80:
        return "largely replicates"
    if ari >= 0.40:
        return "partially replicates"
    return "does not replicate"


def run() -> None:
    _progress("=== I7 - Gower + hierarchical clustering ===")
    df = _load_fixed_subsample()
    numeric, categorical = _prepare_gower_inputs(df)
    _progress(
        f"Governed Gower inputs: numerical={len(CLUSTER_NUMERICAL)} {CLUSTER_NUMERICAL}; "
        f"categorical={len(CLUSTER_CATEGORICAL)} {CLUSTER_CATEGORICAL}"
    )

    distance, used_features, skipped_numeric = _build_gower_distance(numeric, categorical)
    _progress(
        f"Gower matrix shape: {distance.shape}; features_used={used_features}; "
        f"skipped_numeric={skipped_numeric}"
    )

    labels_ik, labels_km = _fit_reference_partitions(df)

    condensed = squareform(distance, checks=False)
    _progress("Fitting average-linkage hierarchical clustering on precomputed Gower distances")
    z_average = linkage(condensed, method="average")
    _progress("Fitting complete-linkage hierarchical clustering on precomputed Gower distances")
    z_complete = linkage(condensed, method="complete")

    average_row, labels_gower = _summarise_linkage(
        z_average,
        distance,
        labels_ik,
        labels_km,
        "average",
        "Primary: average linkage supports arbitrary precomputed Gower distances; Ward not used.",
    )
    complete_row, _ = _summarise_linkage(
        z_complete,
        distance,
        labels_ik,
        labels_km,
        "complete",
        "Robustness note: complete linkage on the same precomputed Gower distances.",
    )

    summary = pd.DataFrame([average_row, complete_row])
    summary.to_csv(SUMMARY_PATH, index=False)
    _progress(f"Saved {SUMMARY_PATH}")

    _plot_dendrogram(z_average)
    _plot_crosstab(labels_gower, labels_ik, average_row["ari_vs_ikmeans"])

    _progress(
        "Average-linkage metrics: "
        f"silhouette={average_row['gower_silhouette']:.4f}; "
        f"ARI vs iK-means={average_row['ari_vs_ikmeans']:.4f}; "
        f"ARI vs k-means={average_row['ari_vs_kmeans']:.4f}"
    )
    _progress(
        "Complete-linkage metrics: "
        f"silhouette={complete_row['gower_silhouette']:.4f}; "
        f"ARI vs iK-means={complete_row['ari_vs_ikmeans']:.4f}; "
        f"ARI vs k-means={complete_row['ari_vs_kmeans']:.4f}"
    )

    phrase = _verdict_phrase(average_row["ari_vs_ikmeans"])
    _progress(
        f"Gower+hierarchical (k={TARGET_K}) vs iK-means headline: "
        f"ARI={average_row['ari_vs_ikmeans']:.3f} -> the 7-segment structure "
        f"{phrase} under a non-Euclidean mixed-data metric."
    )
    _progress("Done.")


def main() -> None:
    if len(sys.argv) > 1:
        _progress("Ignoring command-line flags: this module always uses the fixed FAST_N/FAST_SEED subsample.")
    run()


if __name__ == "__main__":
    main()
