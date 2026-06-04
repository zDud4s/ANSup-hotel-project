"""Second-stage clustering after removing the iKMeans anomaly cluster.

The selected full-run solution (iKMeans + RobustScaler) mostly separates a
tiny anomalous group from the main booking population. This module removes
that smallest iKMeans cluster and reclusters the remaining population with
MiniBatchKMeans to search for more balanced, business-readable segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from ..data.validate import load_raw
from ..evaluation.metrics import compute_indices, mean_pairwise_ari
from ..preprocessing.feature_config import CLUSTER_NUMERICAL, FAST_MODE, FAST_N, FAST_SEED, SEEDS
from ..preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)
from ..clustering.ikmeans import fit_ikmeans
from ..clustering.run_baseline import (
    SILHOUETTE_SAMPLE_FAST,
    SILHOUETTE_SAMPLE_FULL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE = list(range(2, 7))
PCA_SAMPLE_N = 20_000
PROFILE_CATEGORICAL = [
    "hotel",
    "customer_type",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "reserved_room_type",
    "meal",
    "country",
]
POSTHOC_NUMERIC = [
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "is_canceled",
]


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_frames(fast: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        _progress(f"[FULL MODE] {len(df):,} rows")
    df = add_cyclic_seasonality(df)
    return split_clustering_and_profiling(df)


def _select_k(summary: pd.DataFrame) -> dict:
    passing = summary[summary["ari_mean"] >= 0.80]
    if passing.empty:
        chosen = summary.sort_values("sil_mean", ascending=False).iloc[0]
        note = "No k passed ARI >= 0.80; selected max silhouette with caveat."
        gate_passed = False
    else:
        chosen = passing.sort_values("sil_mean", ascending=False).iloc[0]
        note = "Selected max silhouette among k with ARI >= 0.80."
        gate_passed = True
    return {
        "method": "MiniBatchKMeans",
        "variant": "robust_main_population",
        "k_star": int(chosen["k"]),
        "sil_mean": float(chosen["sil_mean"]),
        "db_mean": float(chosen["db_mean"]),
        "ari_mean": float(chosen["ari_mean"]),
        "gate_passed": gate_passed,
        "notes": note,
    }


def _profile_frame(x_input: pd.DataFrame, profiling_frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = x_input.copy()
    for col in profiling_frame.columns:
        if col not in frame.columns:
            frame[col] = profiling_frame[col]
    frame["cluster"] = labels
    return frame


def _profile_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in CLUSTER_NUMERICAL + POSTHOC_NUMERIC if c in frame.columns]
    rows: list[dict] = []
    total = len(frame)
    for cluster, sub in frame.groupby("cluster"):
        row = {"cluster": int(cluster), "n": len(sub), "share": len(sub) / total}
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_median"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


def _profile_categories(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster, sub in frame.groupby("cluster"):
        for feature in PROFILE_CATEGORICAL:
            if feature not in sub.columns:
                continue
            values = sub[feature].fillna("Unknown")
            shares = values.value_counts(normalize=True).head(top_n)
            for value, share in shares.items():
                rows.append({
                    "cluster": int(cluster),
                    "feature": feature,
                    "value": value,
                    "share": share,
                    "n": int((values == value).sum()),
                })
    return pd.DataFrame(rows).sort_values(["cluster", "feature", "share"], ascending=[True, True, False])


def _plot_pca(x: np.ndarray, labels: np.ndarray, selection: dict) -> None:
    sample_n = min(PCA_SAMPLE_N, len(x))
    sample_idx = np.sort(np.random.default_rng(FAST_SEED).choice(len(x), size=sample_n, replace=False))
    x_sample = x[sample_idx]
    labels_sample = labels[sample_idx]

    pca = PCA(n_components=2, random_state=FAST_SEED)
    coords = pca.fit_transform(x_sample)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    cmap = plt.get_cmap("tab10")
    for cluster in sorted(np.unique(labels_sample)):
        mask = labels_sample == cluster
        share = (labels == cluster).mean()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.25,
                   linewidths=0, color=cmap(cluster % 10),
                   label=f"cluster {cluster} ({share:.1%})")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title(f"Main-population MiniBatchKMeans k={selection['k_star']} projected with PCA",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.text(0.01, 0.01,
             "Outlier cluster removed first; PCA is diagnostic and shows only two linear directions.",
             fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig(FIGURES_DIR / "task2_main_population_kmeans_pca.png", dpi=150, bbox_inches="tight")
    plt.close()


def _write_report(selection: dict,
                  anomaly_counts: dict,
                  numeric_profile: pd.DataFrame,
                  top_categories: pd.DataFrame) -> None:
    lines = [
        "# Main-Population Clustering",
        "",
        "Second-stage analysis after removing the smallest iKMeans + RobustScaler cluster.",
        "",
        "## Removed Anomaly Group",
        "",
        f"- Removed cluster label: `{anomaly_counts['removed_label']}`",
        f"- Removed rows: `{anomaly_counts['removed_n']:,}` ({anomaly_counts['removed_share']:.2%})",
        f"- Main-population rows: `{anomaly_counts['main_n']:,}` ({anomaly_counts['main_share']:.2%})",
        "",
        "## Selected Main-Population Solution",
        "",
        f"- Method: `{selection['method']}`",
        "- Scaler: `RobustScaler`",
        f"- Selected k: `{selection['k_star']}`",
        f"- Silhouette mean: `{selection['sil_mean']:.4f}`",
        f"- Davies-Bouldin mean: `{selection['db_mean']:.4f}`",
        f"- Mean ARI: `{selection['ari_mean']:.4f}`",
        f"- Stability gate passed: `{selection['gate_passed']}`",
        f"- Note: {selection['notes']}",
        "",
        "## Segment Size And Numeric Signals",
        "",
        "| Cluster | n | Share | lead_time mean | ADR mean | cancellation rate | special requests mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in numeric_profile.itertuples(index=False):
        lines.append(
            f"| {row.cluster} | {row.n:,} | {row.share:.2%} | "
            f"{getattr(row, 'lead_time_mean'):.2f} | "
            f"{getattr(row, 'adr_mean'):.2f} | "
            f"{getattr(row, 'is_canceled_mean'):.2%} | "
            f"{getattr(row, 'total_of_special_requests_mean'):.2f} |"
        )

    lines.extend([
        "",
        "## Top Categorical Signals",
        "",
    ])
    for cluster in sorted(top_categories["cluster"].unique()):
        lines.append(f"### Cluster {cluster}")
        sub = top_categories[top_categories["cluster"] == cluster]
        for feature in PROFILE_CATEGORICAL:
            values = sub[sub["feature"] == feature].head(3)
            if values.empty:
                continue
            rendered = ", ".join(f"{r.value} ({r.share:.1%})" for r in values.itertuples(index=False))
            lines.append(f"- `{feature}`: {rendered}")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "This second-stage result should be used to decide whether there are more meaningful booking segments inside the dominant population after the tiny anomaly group is removed.",
        "",
    ])
    (REPORT_DIR / "task2_main_population_clustering.md").write_text("\n".join(lines), encoding="utf-8")


def run(fast: bool, top_n: int) -> None:
    _progress("=== Main-population clustering ===")
    sil_sample = SILHOUETTE_SAMPLE_FAST if fast else SILHOUETTE_SAMPLE_FULL
    x_input, profiling_frame = _load_frames(fast)

    _progress("Fitting RobustScaler preprocessing")
    preproc = build_preprocessor("robust")
    x_all = preproc.fit_transform(x_input)
    _progress(f"X shape: {x_all.shape}")

    _progress("Identifying smallest iKMeans cluster")
    anomaly_labels, _, k_auto = fit_ikmeans(x_all, seed=SEEDS[0])
    counts = pd.Series(anomaly_labels).value_counts().sort_index()
    removed_label = int(counts.idxmin())
    main_mask = anomaly_labels != removed_label
    anomaly_counts = {
        "removed_label": removed_label,
        "removed_n": int((~main_mask).sum()),
        "removed_share": float((~main_mask).mean()),
        "main_n": int(main_mask.sum()),
        "main_share": float(main_mask.mean()),
        "initial_k": int(k_auto),
    }
    _progress(
        f"Removed cluster {removed_label}: {anomaly_counts['removed_n']:,} rows "
        f"({anomaly_counts['removed_share']:.2%})"
    )

    x_main = x_all[main_mask]
    x_input_main = x_input.loc[main_mask].reset_index(drop=True)
    profiling_main = profiling_frame.loc[main_mask].reset_index(drop=True)

    rows: list[dict] = []
    labels_cache: dict[int, dict[int, np.ndarray]] = {}
    total_runs = len(K_RANGE) * len(SEEDS)
    run_no = 0
    for k in K_RANGE:
        _progress(f"  [main k-means] starting k={k}")
        labels_by_seed: dict[int, np.ndarray] = {}
        for seed in SEEDS:
            run_no += 1
            _progress(f"  [main k-means {run_no}/{total_runs}] fit k={k} seed={seed}")
            mb = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                                 batch_size=1024, max_iter=300)
            labels = mb.fit_predict(x_main)
            labels_by_seed[seed] = labels
            _progress(f"  [main k-means {run_no}/{total_runs}] metrics k={k} seed={seed}")
            idx = compute_indices(x_main, labels, silhouette_sample_size=sil_sample, seed=seed)
            rows.append({
                "method": "MiniBatchKMeans",
                "variant": "robust_main_population",
                "k": k,
                "seed": seed,
                "silhouette": idx["silhouette"],
                "calinski_harabasz": idx["calinski_harabasz"],
                "davies_bouldin": idx["davies_bouldin"],
                "ari_vs_seed0": adjusted_rand_score(labels_by_seed[SEEDS[0]], labels),
            })
            _progress(
                f"  [main k-means {run_no}/{total_runs}] done "
                f"sil={idx['silhouette']:.4f} db={idx['davies_bouldin']:.4f}"
            )
        labels_cache[k] = labels_by_seed

    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["method", "variant", "k"])
               .agg(sil_mean=("silhouette", "mean"),
                    sil_std=("silhouette", "std"),
                    ch_mean=("calinski_harabasz", "mean"),
                    ch_std=("calinski_harabasz", "std"),
                    db_mean=("davies_bouldin", "mean"),
                    db_std=("davies_bouldin", "std"))
               .reset_index())
    summary["ari_mean"] = summary["k"].map(lambda k: mean_pairwise_ari(labels_cache[int(k)]))
    selection = _select_k(summary)

    raw.to_csv(TABLES_DIR / "task2_main_population_kmeans_runs.csv", index=False)
    summary.to_csv(TABLES_DIR / "task2_main_population_kmeans_summary.csv", index=False)
    pd.DataFrame([selection]).to_csv(TABLES_DIR / "task2_main_population_kmeans_selection.csv", index=False)

    selected_labels = labels_cache[selection["k_star"]][SEEDS[0]]
    profile = _profile_frame(x_input_main, profiling_main, selected_labels)
    numeric_profile = _profile_numeric(profile)
    top_categories = _profile_categories(profile, top_n=top_n)
    numeric_profile.to_csv(TABLES_DIR / "task2_main_population_cluster_overview.csv", index=False)
    top_categories.to_csv(TABLES_DIR / "task2_main_population_cluster_top_categories.csv", index=False)
    _plot_pca(x_main, selected_labels, selection)
    _write_report(selection, anomaly_counts, numeric_profile, top_categories)

    _progress(f"Selected k={selection['k_star']} with sil={selection['sil_mean']:.4f}, "
              f"ARI={selection['ari_mean']:.4f}")
    _progress("Saved main-population tables, report, and PCA figure.")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    fast = FAST_MODE
    if args.full:
        fast = False
    elif args.fast:
        fast = True
    run(fast=fast, top_n=args.top_n)


if __name__ == "__main__":
    main()
