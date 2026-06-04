"""Profile the selected iKMeans + RobustScaler clustering solution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.validate import load_raw
from ..preprocessing.feature_config import CLUSTER_NUMERICAL, FAST_MODE, FAST_N, FAST_SEED
from ..preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)
from ..clustering.ikmeans import fit_ikmeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_PROFILE = [
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
    "previous_cancellations",
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


def _build_profile_frame(x_input: pd.DataFrame, profiling_frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = x_input.copy()
    for col in profiling_frame.columns:
        if col not in frame.columns:
            frame[col] = profiling_frame[col]
    frame["cluster"] = labels
    return frame


def _overview(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in CLUSTER_NUMERICAL + POSTHOC_NUMERIC if c in frame.columns]
    rows: list[dict] = []
    total = len(frame)
    for cluster, sub in frame.groupby("cluster"):
        row = {
            "cluster": int(cluster),
            "n": len(sub),
            "share": len(sub) / total,
        }
        for col in numeric_cols:
            row[f"{col}_mean"] = pd.to_numeric(sub[col], errors="coerce").mean()
            row[f"{col}_median"] = pd.to_numeric(sub[col], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


def _top_categories(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster, sub in frame.groupby("cluster"):
        for feature in CATEGORICAL_PROFILE:
            if feature not in sub.columns:
                continue
            shares = sub[feature].fillna("Unknown").value_counts(normalize=True).head(top_n)
            for value, share in shares.items():
                rows.append({
                    "cluster": int(cluster),
                    "feature": feature,
                    "value": value,
                    "share": share,
                    "n": int((sub[feature].fillna("Unknown") == value).sum()),
                })
    return pd.DataFrame(rows).sort_values(["cluster", "feature", "share"], ascending=[True, True, False])


def _write_report(overview: pd.DataFrame, top_categories: pd.DataFrame, k_auto: int) -> None:
    lines = [
        "# iKMeans Robust Cluster Profile",
        "",
        "Selected solution: `iKMeans + RobustScaler`.",
        "",
        f"Auto-determined `k`: `{k_auto}`.",
        "",
        "## Cluster Sizes",
        "",
        "| Cluster | n | Share |",
        "|---:|---:|---:|",
    ]
    for row in overview.itertuples(index=False):
        lines.append(f"| {row.cluster} | {row.n:,} | {row.share:.2%} |")

    lines.extend([
        "",
        "## Numeric Signals",
        "",
        "| Cluster | lead_time mean | previous_cancellations mean | adr mean | cancellation rate | special requests mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in overview.itertuples(index=False):
        lines.append(
            "| "
            f"{row.cluster} | "
            f"{getattr(row, 'lead_time_mean'):.2f} | "
            f"{getattr(row, 'previous_cancellations_mean'):.2f} | "
            f"{getattr(row, 'adr_mean'):.2f} | "
            f"{getattr(row, 'is_canceled_mean'):.2%} | "
            f"{getattr(row, 'total_of_special_requests_mean'):.2f} |"
        )

    lines.extend([
        "",
        "## Top Categorical Values",
        "",
        "Top values are reported within each cluster. `country` is descriptive only and should be interpreted as market-of-origin, not as a substantive causal explanation.",
        "",
    ])
    for cluster in sorted(top_categories["cluster"].unique()):
        lines.append(f"### Cluster {cluster}")
        sub = top_categories[top_categories["cluster"] == cluster]
        for feature in CATEGORICAL_PROFILE:
            vals = sub[sub["feature"] == feature].head(3)
            if vals.empty:
                continue
            rendered = ", ".join(f"{r.value} ({r.share:.1%})" for r in vals.itertuples(index=False))
            lines.append(f"- `{feature}`: {rendered}")
        lines.append("")

    lines.extend([
        "## Interpretation Prompt",
        "",
        "Use this profile to decide whether the small cluster is a meaningful business segment or mainly an anomaly/outlier group. iK-means discovers and refits entirely in the full governed feature space (no PCA); the diagnostic PCA scatter is only a 2-D viewing aid. This table explains which variables actually drive the split.",
        "",
    ])
    (REPORT_DIR / "task2_ikmeans_robust_cluster_profile.md").write_text("\n".join(lines), encoding="utf-8")


def run(fast: bool, top_n: int, seed: int) -> None:
    _progress("=== iKMeans robust cluster profiling ===")
    x_input, profiling_frame = _load_frames(fast)
    preproc = build_preprocessor("robust")
    x = preproc.fit_transform(x_input)
    _progress(f"X shape: {x.shape}")

    labels, _, k_auto = fit_ikmeans(x, seed=seed)
    frame = _build_profile_frame(x_input, profiling_frame, labels)

    overview = _overview(frame)
    top_categories = _top_categories(frame, top_n=top_n)

    overview_path = TABLES_DIR / "task2_ikmeans_robust_cluster_overview.csv"
    cat_path = TABLES_DIR / "task2_ikmeans_robust_cluster_top_categories.csv"
    overview.to_csv(overview_path, index=False)
    top_categories.to_csv(cat_path, index=False)
    _write_report(overview, top_categories, k_auto)

    _progress(f"Saved overview: {overview_path}")
    _progress(f"Saved top categories: {cat_path}")
    _progress(f"Saved report: {REPORT_DIR / 'task2_ikmeans_robust_cluster_profile.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fast = FAST_MODE
    if args.full:
        fast = False
    elif args.fast:
        fast = True

    run(fast=fast, top_n=args.top_n, seed=args.seed)


if __name__ == "__main__":
    main()
