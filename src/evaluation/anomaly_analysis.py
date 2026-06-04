"""Extension Module E1 - cluster-aware anomaly analysis.

Scores each booking by distance to its own iK-means cluster centre, normalised
by that cluster's median distance. The headline analysis uses the canonical
StandardScaler matrix; sensitivity checks compare against the RobustScaler
tail cluster and the E4 PCA representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..clustering.ikmeans import fit_ikmeans
from ..data.validate import load_raw
from ..preprocessing.feature_config import FAST_MODE, FAST_N, FAST_SEED, SEEDS
from ..preprocessing.pipeline import (
    add_booking_features,
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)

try:
    from .pca_study import fit_pca_projection
except ImportError:  # pragma: no cover - fallback for direct partial imports
    fit_pca_projection = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
TOP_AUDIT_N = 20
TOP_OVERLAP_N = 200
PCA_SCATTER_SAMPLE_N = 8_000

AUDIT_COLUMNS = [
    "lead_time",
    "total_nights",
    "adr",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "country",
    "is_canceled",
]

WHY_NUMERIC_COLUMNS = [
    "lead_time",
    "total_nights",
    "party_size",
    "weekend_share",
    "adr",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

COUNT_COLUMNS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "booking_changes",
    "days_in_waiting_list",
]


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_aligned_frames(fast: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        df = df.copy().reset_index(drop=True)
        _progress(f"[FULL MODE] {len(df):,} rows")

    df = add_cyclic_seasonality(df)
    x_input, profiling_frame = split_clustering_and_profiling(df)
    audit_frame = add_booking_features(df)
    return x_input, profiling_frame.reset_index(drop=True), audit_frame.reset_index(drop=True)


def _build_standard_matrix(x_input: pd.DataFrame) -> np.ndarray:
    preproc = build_preprocessor(StandardScaler)
    x = preproc.fit_transform(x_input)
    _progress(f"StandardScaler X shape: {x.shape}")
    return x


def _score_against_centres(
    x: np.ndarray,
    labels: np.ndarray,
    centres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(x - centres[labels], axis=1)
    scores = np.empty_like(distances, dtype=float)
    for cluster in np.unique(labels):
        mask = labels == cluster
        median_distance = float(np.median(distances[mask]))
        scores[mask] = distances[mask] / (median_distance + EPS)
    return scores, distances


def _combine_profile(
    x_input: pd.DataFrame,
    profiling_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = x_input.reset_index(drop=True).copy()
    for col in profiling_frame.columns:
        if col not in frame.columns:
            frame[col] = profiling_frame[col].reset_index(drop=True)
    for col in audit_frame.columns:
        if col not in frame.columns:
            frame[col] = audit_frame[col].reset_index(drop=True)
    return frame


def _data_quality_flags(frame: pd.DataFrame) -> pd.Series:
    flags = pd.Series(False, index=frame.index)
    if "adr" in frame.columns:
        flags |= pd.to_numeric(frame["adr"], errors="coerce").le(0).fillna(False)

    if {"adults", "children", "babies"}.issubset(frame.columns):
        adults = pd.to_numeric(frame["adults"], errors="coerce").fillna(0)
        children = pd.to_numeric(frame["children"], errors="coerce").fillna(0)
        babies = pd.to_numeric(frame["babies"], errors="coerce").fillna(0)
        flags |= adults.eq(0) & children.eq(0) & babies.eq(0)

    for col in COUNT_COLUMNS:
        if col in frame.columns:
            flags |= pd.to_numeric(frame[col], errors="coerce").lt(0).fillna(False)
    return flags.astype(bool)


def _why_strings(frame: pd.DataFrame, labels: np.ndarray, top_idx: np.ndarray) -> list[str]:
    numeric_cols = [c for c in WHY_NUMERIC_COLUMNS if c in frame.columns]
    numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")

    cluster_medians = numeric.groupby(labels).median()
    cluster_mads = numeric.groupby(labels).apply(lambda sub: (sub - sub.median()).abs().median())

    reasons: list[str] = []
    for idx in top_idx:
        cluster = int(labels[idx])
        row = numeric.iloc[idx]
        med = cluster_medians.loc[cluster]
        mad = cluster_mads.loc[cluster].replace(0, np.nan)
        deviation = ((row - med).abs() / (mad + EPS)).replace([np.inf, -np.inf], np.nan)
        best = deviation.dropna().sort_values(ascending=False).head(3)
        parts = []
        for col in best.index:
            value = row[col]
            median_value = med[col]
            if pd.notna(value) and pd.notna(median_value):
                parts.append(f"{col}={value:g} vs cluster median {median_value:g}")
        reasons.append("; ".join(parts) if parts else "No large numeric profiling deviation available")
    return reasons


def _top20_table(
    profile: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    top_idx = np.argsort(scores)[::-1][:TOP_AUDIT_N]
    flags = _data_quality_flags(profile)

    rows: list[dict] = []
    why = _why_strings(profile, labels, top_idx)
    for rank, idx in enumerate(top_idx, start=1):
        row = {
            "rank": rank,
            "row_index": int(idx),
            "cluster": int(labels[idx]),
            "score": float(scores[idx]),
            "data_quality_flag": bool(flags.iloc[idx]),
            "why": why[rank - 1],
        }
        for col in AUDIT_COLUMNS:
            if col in profile.columns:
                row[col] = profile.iloc[idx][col]
        rows.append(row)
    return pd.DataFrame(rows)


def _jaccard(a: set[int], b: set[int]) -> float:
    union = len(a | b)
    return 0.0 if union == 0 else len(a & b) / union


def _fit_pca_90(x: np.ndarray) -> tuple[np.ndarray, int]:
    if fit_pca_projection is not None:
        _, x_pca, _, _, n_keep = fit_pca_projection(x)
        return x_pca, int(n_keep)

    pca = PCA(svd_solver="full", random_state=SEEDS[0])
    x_all = pca.fit_transform(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
    return x_all[:, :n_keep], n_keep


def _sensitivity(
    x_input: pd.DataFrame,
    x_standard: np.ndarray,
    standard_top: set[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    robust_x = build_preprocessor("robust").fit_transform(x_input)
    robust_labels, _, robust_k = fit_ikmeans(robust_x, seed=SEEDS[0])
    robust_counts = pd.Series(robust_labels).value_counts().sort_values()
    smallest_label = int(robust_counts.index[0])
    robust_tail = set(np.flatnonzero(robust_labels == smallest_label).tolist())
    _progress(
        "RobustScaler iK-means cluster sizes: "
        + ", ".join(f"{int(label)}={int(count)}" for label, count in pd.Series(robust_labels).value_counts().sort_index().items())
    )
    _progress(f"Smallest RobustScaler cluster: {smallest_label} (k={robust_k}, n={len(robust_tail):,})")

    x_pca, n_keep = _fit_pca_90(x_standard)
    pca_labels, pca_centres, pca_k = fit_ikmeans(x_pca, seed=SEEDS[0])
    pca_scores, _ = _score_against_centres(x_pca, pca_labels, pca_centres)
    pca_top = set(np.argsort(pca_scores)[::-1][:min(TOP_OVERLAP_N, len(pca_scores))].tolist())
    _progress(f"PCA-space iK-means k={pca_k}, n_keep={n_keep}")

    rows = [
        {
            "comparison": "Standard top-N vs Robust smallest cluster",
            "n_top": min(TOP_OVERLAP_N, len(x_standard)),
            "overlap_count": len(standard_top & robust_tail),
            "jaccard": _jaccard(standard_top, robust_tail),
        },
        {
            "comparison": "Original-space top-N vs PCA-space top-N",
            "n_top": min(TOP_OVERLAP_N, len(x_standard)),
            "overlap_count": len(standard_top & pca_top),
            "jaccard": _jaccard(standard_top, pca_top),
        },
    ]
    return pd.DataFrame(rows), pca_scores, x_pca, n_keep


def _plot_score_by_cluster(labels: np.ndarray, scores: np.ndarray) -> Path:
    clusters = sorted(np.unique(labels))
    data = [scores[labels == cluster] for cluster in clusters]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.boxplot(data, tick_labels=[str(c) for c in clusters], showfliers=True)
    ax.set_xlabel("iK-means cluster")
    ax.set_ylabel("Cluster-conditioned anomaly score")
    ax.set_title("E1 anomaly score distribution by headline StandardScaler cluster")
    ax.grid(alpha=0.25, axis="y")
    path = FIGURES_DIR / "e1_anomaly_score_by_cluster.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _plot_pca_scatter(x: np.ndarray, scores: np.ndarray, top20: pd.DataFrame) -> Path:
    sample_n = min(PCA_SCATTER_SAMPLE_N, len(x))
    rng = np.random.default_rng(FAST_SEED)
    sample_idx = np.sort(rng.choice(len(x), size=sample_n, replace=False))
    top_idx = top20["row_index"].to_numpy(dtype=int)
    plot_idx = np.unique(np.concatenate([sample_idx, top_idx]))

    coords = PCA(n_components=2, random_state=SEEDS[0]).fit_transform(x[plot_idx])
    plot_scores = scores[plot_idx]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=plot_scores,
        s=10 + 18 * np.clip(plot_scores / np.nanpercentile(scores, 99), 0, 1),
        cmap="viridis",
        alpha=0.45,
        linewidths=0,
    )
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(plot_idx)}
    top_positions = [idx_to_pos[int(idx)] for idx in top_idx if int(idx) in idx_to_pos]
    ax.scatter(
        coords[top_positions, 0],
        coords[top_positions, 1],
        facecolors="none",
        edgecolors="#D62728",
        s=95,
        linewidths=1.4,
        label="top-20 anomalies",
    )
    for row in top20.itertuples(index=False):
        pos = idx_to_pos.get(int(row.row_index))
        if pos is not None:
            ax.text(coords[pos, 0], coords[pos, 1], str(int(row.rank)), fontsize=7, color="#B22222")
    ax.set_xlabel("PC1 diagnostic projection")
    ax.set_ylabel("PC2 diagnostic projection")
    ax.set_title("E1 top anomalies on 2D PCA diagnostic projection")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cluster-conditioned anomaly score")
    path = FIGURES_DIR / "e1_anomaly_pca_scatter.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _plot_overlap(overlap: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels = ["Std vs Robust tail", "Original vs PCA"]
    bars = ax.bar(labels, overlap["jaccard"], color=["#4C78A8", "#F58518"], alpha=0.85)
    for bar, row in zip(bars, overlap.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{row.overlap_count}/{row.n_top}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(0.12, float(overlap["jaccard"].max()) * 1.25))
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("E1 sensitivity overlap for predefined top-N anomaly sets")
    ax.grid(alpha=0.25, axis="y")
    path = FIGURES_DIR / "e1_sensitivity_overlap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run(fast: bool = FAST_MODE) -> None:
    _progress("=== E1 - cluster-aware anomaly analysis ===")
    x_input, profiling_frame, audit_frame = _load_aligned_frames(fast)
    profile = _combine_profile(x_input, profiling_frame, audit_frame)

    x_standard = _build_standard_matrix(x_input)
    labels, centres, k_auto = fit_ikmeans(x_standard, seed=SEEDS[0])
    _progress(f"Headline StandardScaler iK-means k={k_auto} (seed={SEEDS[0]})")

    scores, _ = _score_against_centres(x_standard, labels, centres)
    top20 = _top20_table(profile, labels, scores)
    flags = _data_quality_flags(profile)
    top_n = min(TOP_OVERLAP_N, len(scores))
    standard_top = set(np.argsort(scores)[::-1][:top_n].tolist())

    overlap, _, _, n_keep = _sensitivity(x_input, x_standard, standard_top)

    top20_path = TABLES_DIR / "e1_top20_anomalies.csv"
    overlap_path = TABLES_DIR / "e1_sensitivity_overlap.csv"
    top20.to_csv(top20_path, index=False)
    overlap.to_csv(overlap_path, index=False)

    fig_score = _plot_score_by_cluster(labels, scores)
    fig_scatter = _plot_pca_scatter(x_standard, scores, top20)
    fig_overlap = _plot_overlap(overlap)

    dq_count = int(flags.sum())
    behaviour_count = int(len(flags) - dq_count)
    top20_dq = int(top20["data_quality_flag"].sum())
    top20_behaviour = int(len(top20) - top20_dq)

    _progress("\nSummary:")
    _progress(
        "  No ground-truth anomaly labels exist; 'rare' is not the same as 'bad'. "
        "Cluster-conditioned scores flag points atypical for their OWN segment, "
        "which is the intended, auditable notion of anomaly here."
    )
    _progress(
        f"  Data-quality vs rare-but-plausible split: all rows "
        f"{dq_count:,} flagged / {behaviour_count:,} rare-but-plausible; "
        f"top-20 {top20_dq:,} flagged / {top20_behaviour:,} rare-but-plausible."
    )
    for row in overlap.itertuples(index=False):
        _progress(
            f"  {row.comparison}: overlap={row.overlap_count}/{row.n_top}, "
            f"Jaccard={row.jaccard:.4f}"
        )
    _progress(
        f"  PCA sensitivity used E4's predefined 90% variance rule (n_keep={n_keep}). "
        f"Top-N for overlap was fixed at N={top_n}; the score definition was not tuned."
    )
    _progress(
        "  iK-means is deterministic under this anomalous-pattern initialisation; "
        "with the fixed seed used for final tie-breaking, the anomaly ranking is "
        "reproducible without seed averaging."
    )

    for path in [top20_path, overlap_path, fig_score, fig_scatter, fig_overlap]:
        _progress(f"Saved {path}")


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast)


if __name__ == "__main__":
    main()
