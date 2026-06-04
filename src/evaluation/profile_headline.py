"""Profile the headline iK-means + StandardScaler clustering solution."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ..clustering.ikmeans import fit_ikmeans
from ..preprocessing.feature_config import (
    CLUSTER_CATEGORICAL,
    CLUSTER_NUMERICAL,
    FAST_MODE,
    SEEDS,
)
from ..preprocessing.pipeline import build_preprocessor
from .profile_ikmeans import _build_profile_frame, _load_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

HEADLINE_NUMERICAL = [
    "lead_time",
    "total_nights",
    "party_size",
    "has_kids",
    "weekend_share",
]
POSTHOC_MEANS = [
    "adr",
    "total_of_special_requests",
    "required_car_parking_spaces",
]
POSTHOC_OUTCOME = "is_canceled"


def _progress(message: str) -> None:
    print(message, flush=True)


def _overview(frame: pd.DataFrame) -> pd.DataFrame:
    total = len(frame)
    rows: list[dict] = []
    for cluster, sub in frame.groupby("cluster"):
        row = {
            "cluster": int(cluster),
            "n": int(len(sub)),
            "share_pct": len(sub) / total * 100,
        }
        for col in HEADLINE_NUMERICAL:
            row[f"{col}_mean"] = pd.to_numeric(sub[col], errors="coerce").mean()
        for col in POSTHOC_MEANS:
            row[f"posthoc_{col}_mean"] = pd.to_numeric(sub[col], errors="coerce").mean()
        row["posthoc_is_canceled_descriptive_rate"] = pd.to_numeric(
            sub[POSTHOC_OUTCOME], errors="coerce"
        ).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def _top_categories(frame: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster, sub in frame.groupby("cluster"):
        for feature in CLUSTER_CATEGORICAL:
            shares = sub[feature].fillna("Unknown").value_counts(normalize=True).head(top_n)
            counts = sub[feature].fillna("Unknown").value_counts().reindex(shares.index)
            for rank, (value, share) in enumerate(shares.items(), start=1):
                rows.append(
                    {
                        "cluster": int(cluster),
                        "feature": feature,
                        "rank": rank,
                        "value": value,
                        "share_pct": share * 100,
                        "n": int(counts.loc[value]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["cluster", "feature", "rank"]).reset_index(drop=True)


def _write_heatmap(overview: pd.DataFrame, path: Path) -> None:
    means = overview.set_index("cluster")[[f"{c}_mean" for c in HEADLINE_NUMERICAL]]
    means.columns = HEADLINE_NUMERICAL
    col_std = means.std(axis=0, ddof=0).replace(0, np.nan)
    z = ((means - means.mean(axis=0)) / col_std).fillna(0.0)

    values = z.to_numpy()
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_title("Headline iK-means + StandardScaler profile: z-scored cluster means")
    ax.set_xlabel("Clustering numeric feature")
    ax.set_ylabel("Cluster")
    ax.set_xticks(np.arange(len(HEADLINE_NUMERICAL)), labels=HEADLINE_NUMERICAL, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(z.index)), labels=[str(c) for c in z.index])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cluster mean z-score across clusters")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(fast: bool) -> None:
    _progress("=== Headline iK-means + StandardScaler cluster profiling ===")
    x_input, profiling_frame = _load_frames(fast)

    missing_inputs = [c for c in HEADLINE_NUMERICAL + CLUSTER_CATEGORICAL if c not in x_input.columns]
    if missing_inputs:
        raise SystemExit(f"Missing clustering/profile inputs: {missing_inputs}")
    missing_posthoc = [c for c in POSTHOC_MEANS + [POSTHOC_OUTCOME] if c not in profiling_frame.columns]
    if missing_posthoc:
        raise SystemExit(f"is_canceled/post-hoc profiling columns absent: {missing_posthoc}")

    preproc = build_preprocessor("standard")
    x = preproc.fit_transform(x_input)
    _progress(f"X shape: {x.shape}")

    seed = SEEDS[0]
    labels, _, k_auto = fit_ikmeans(x, seed=seed)
    _progress(f"Auto-k: {k_auto} (seed={seed}, scaler=standard)")
    if k_auto != 7:
        raise SystemExit(f"StandardScaler iK-means auto-k was {k_auto}, not approximately 7.")

    frame = _build_profile_frame(x_input, profiling_frame, labels)
    overview = _overview(frame)
    top_categories = _top_categories(frame, top_n=3)

    overview_path = TABLES_DIR / "headline_k7_overview.csv"
    top_categories_path = TABLES_DIR / "headline_k7_top_categories.csv"
    heatmap_path = FIGURES_DIR / "headline_k7_profile_heatmap.png"

    overview.to_csv(overview_path, index=False)
    top_categories.to_csv(top_categories_path, index=False)
    _write_heatmap(overview, heatmap_path)

    _progress("Overview table:")
    _progress(overview.to_string(index=False))
    _progress(f"Saved overview: {overview_path}")
    _progress(f"Saved top categories: {top_categories_path}")
    _progress(f"Saved heatmap: {heatmap_path}")


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast)


if __name__ == "__main__":
    main()
