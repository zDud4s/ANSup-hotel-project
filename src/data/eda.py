"""Reproducible EDA figure generator.

Regenerates the four exploratory figures that previously existed only as
outputs of notebooks/task1_notebook.ipynb, so run_all can rebuild them
with no manual steps. Figure-only: reads the validated raw CSV via the
shared loader and writes PNGs to figures/. No clustering, no governance
logic, no seaborn dependency.

Outputs:
    figures/eda_missing.png
    figures/eda_numerical_distributions.png
    figures/eda_categorical_distributions.png
    figures/eda_correlation_matrix.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .validate import load_raw
from ..preprocessing.feature_config import FAST_MODE, FAST_N, FAST_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PLOT_NUMERICAL = [
    "lead_time", "adr", "stays_in_week_nights", "stays_in_weekend_nights",
    "adults", "children", "booking_changes", "days_in_waiting_list",
]
PLOT_CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "deposit_type", "customer_type", "reserved_room_type",
]


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_df(fast: bool) -> pd.DataFrame:
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        _progress(f"[FULL MODE] {len(df):,} rows")
    return df


def plot_missing(df: pd.DataFrame) -> None:
    missing = (
        df.isnull().sum()
        .rename("n_missing")
        .to_frame()
        .assign(pct=lambda x: (x["n_missing"] / len(df) * 100).round(2))
        .query("n_missing > 0")
        .sort_values("pct", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 2.5))
    if not missing.empty:
        missing["pct"].plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    plt.tight_layout()
    out = FIGURES_DIR / "eda_missing.png"
    plt.savefig(out)
    plt.close()
    _progress(f"Saved {out}")


def plot_numerical(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for ax, col in zip(axes.flat, PLOT_NUMERICAL):
        df[col].dropna().plot.hist(bins=40, ax=ax, color="steelblue", edgecolor="none")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    plt.suptitle("Numerical feature distributions", fontsize=11, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "eda_numerical_distributions.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    _progress(f"Saved {out}")


def plot_categorical(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, col in zip(axes.flat, PLOT_CATEGORICAL):
        df[col].value_counts().head(10).plot.barh(ax=ax, color="steelblue")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("count", fontsize=8)
    axes.flat[-1].set_visible(False)
    plt.suptitle("Categorical distributions (top 10 per feature)", fontsize=11, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "eda_categorical_distributions.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    _progress(f"Saved {out}")


def plot_correlation(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    data = np.ma.masked_array(corr.values, mask=mask)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(num_cols, fontsize=7)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix - numerical features")
    plt.tight_layout()
    out = FIGURES_DIR / "eda_correlation_matrix.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    _progress(f"Saved {out}")


def run(fast: bool) -> None:
    _progress("=== EDA figure generation ===")
    df = _load_df(fast)
    plot_missing(df)
    plot_numerical(df)
    plot_categorical(df)
    plot_correlation(df)
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
