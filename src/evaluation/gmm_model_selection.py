"""Task 2 / RQ5 — extended-k GMM BIC/AIC trajectory diagnostic.

This module uses the same StandardScaler representation as the canonical
``src.clustering.run_gaussian`` GMM: raw data loading, governance split, then
``build_preprocessor(StandardScaler).fit_transform(df_input)``. It deliberately
does not append to experiments.csv or change the canonical {2..8} GMM results.

The diagnostic fits full-covariance GaussianMixture models over a predefined
extended k range with one fixed seed and ``n_init=3``. That is sufficient for a
reproducible trajectory-shape check: the goal is to test whether the original
comparison bound masked an interior BIC optimum, not to replace the canonical
multi-seed GMM experiment.

Outputs:
- tables/gmm_model_selection_extended.csv
- tables/gmm_model_selection_verdict.csv
- figures/gmm_model_selection_extended.png
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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..clustering.run_baseline import load_clustering_input
from ..preprocessing.feature_config import FAST_MODE, SEEDS
from ..preprocessing.pipeline import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXTENDED_K_RANGE = list(range(2, 16))  # extended beyond the {2..8} comparison range to test whether BIC identifies an interior optimum or keeps decreasing (the latter => no natural Gaussian component count for this representation).
COMPARISON_K_RANGE = list(range(2, 9))
MONOTONE_TOL = 1e-6


def _progress(message: str) -> None:
    print(message, flush=True)


def build_standard_representation(fast: bool) -> np.ndarray:
    df_input = load_clustering_input(fast)
    preproc = build_preprocessor(StandardScaler)
    X = preproc.fit_transform(df_input)
    return X


def fit_trajectory(X: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for k in EXTENDED_K_RANGE:
        _progress(f"[GMM selection] fitting k={k}")
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=SEEDS[0],
            n_init=3,
            max_iter=200,
        )
        try:
            gmm.fit(X)
            row = {
                "k": k,
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
                "converged": bool(gmm.converged_),
            }
            _progress(
                f"[GMM selection] k={k} bic={row['bic']:.2f} "
                f"aic={row['aic']:.2f} converged={row['converged']}"
            )
        except Exception as exc:
            row = {
                "k": k,
                "bic": np.nan,
                "aic": np.nan,
                "converged": False,
            }
            _progress(f"[GMM selection] k={k} failed: {type(exc).__name__}: {exc}")
        rows.append(row)
    return pd.DataFrame(rows)


def build_verdict(trajectory: pd.DataFrame) -> dict:
    valid = trajectory.dropna(subset=["bic"]).copy()
    if valid.empty:
        justification = (
            "No valid BIC values were produced in the predefined extended range; "
            "the trajectory diagnostic cannot identify a component-count optimum."
        )
        return {
            "k_star_extended": np.nan,
            "k_star_comparison": np.nan,
            "monotone": False,
            "status": "no_valid_bic",
            "justification": justification,
        }

    k_star_extended = int(valid.loc[valid["bic"].idxmin(), "k"])
    comparison = valid[valid["k"].isin(COMPARISON_K_RANGE)]
    k_star_comparison = (
        int(comparison.loc[comparison["bic"].idxmin(), "k"])
        if not comparison.empty
        else np.nan
    )

    has_all_k = list(valid["k"].astype(int)) == EXTENDED_K_RANGE
    bic_values = valid.sort_values("k")["bic"].to_numpy(dtype=float)
    monotone = bool(has_all_k and np.all(np.diff(bic_values) <= MONOTONE_TOL))
    comparison_bound_was_binding = bool(
        not pd.isna(k_star_comparison)
        and int(k_star_comparison) == COMPARISON_K_RANGE[-1]
        and k_star_extended > COMPARISON_K_RANGE[-1]
    )

    if monotone:
        status = "monotone"
        justification = (
            "BIC decreases monotonically through k=15; no interior optimum within "
            "the justified extended range => GMM finds no natural component count, "
            "consistent with a poor Gaussian fit for this mixed-type representation."
        )
    elif EXTENDED_K_RANGE[0] < k_star_extended < EXTENDED_K_RANGE[-1]:
        status = "interior_optimum"
        binding = "was" if comparison_bound_was_binding else "was not"
        justification = (
            f"BIC attains an interior minimum at k={k_star_extended}; the original "
            f"{{2..8}} bound {binding} binding."
        )
    else:
        status = "boundary_optimum"
        binding = "was" if comparison_bound_was_binding else "was not"
        justification = (
            f"BIC attains its extended-range minimum at boundary k={k_star_extended}; "
            f"no interior optimum was found in {{2..15}}, and the original {{2..8}} "
            f"bound {binding} binding."
        )

    failed_k = trajectory.loc[trajectory["bic"].isna(), "k"].astype(int).tolist()
    if failed_k:
        justification += f" Failed fits recorded for k={failed_k}."

    return {
        "k_star_extended": k_star_extended,
        "k_star_comparison": k_star_comparison,
        "monotone": monotone,
        "status": status,
        "justification": justification,
    }


def plot_trajectory(trajectory: pd.DataFrame, verdict: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = trajectory.sort_values("k")

    ax.axvspan(EXTENDED_K_RANGE[0], COMPARISON_K_RANGE[-1], color="#d8e8ff", alpha=0.35,
               label="original comparison range {2..8}")
    ax.axvline(COMPARISON_K_RANGE[-1], color="#4a6fa5", linestyle="--", linewidth=1.2,
               label="original bound k=8")
    ax.plot(plot_df["k"], plot_df["bic"], marker="o", linewidth=1.8, label="BIC")
    ax.plot(plot_df["k"], plot_df["aic"], marker="s", linewidth=1.8, label="AIC")

    k_star = verdict["k_star_extended"]
    if not pd.isna(k_star):
        star_row = plot_df[plot_df["k"] == int(k_star)].iloc[0]
        ax.scatter([star_row["k"]], [star_row["bic"]], s=95, marker="*", color="#b00020",
                   zorder=5, label=f"BIC minimum k={int(k_star)}")

    title_status = {
        "monotone": "monotone BIC decrease",
        "interior_optimum": f"interior BIC optimum at k={int(k_star)}",
        "boundary_optimum": f"boundary BIC optimum at k={int(k_star)}",
    }.get(str(verdict["status"]), str(verdict["status"]))
    ax.set_title(f"GMM extended-k BIC/AIC trajectory: {title_status}")
    ax.set_xlabel("Number of Gaussian components (k)")
    ax.set_ylabel("Information criterion (lower is better)")
    ax.set_xticks(EXTENDED_K_RANGE)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gmm_model_selection_extended.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(fast: bool = FAST_MODE) -> None:
    _progress("=== GMM extended-k BIC/AIC trajectory diagnostic ===")
    X = build_standard_representation(fast)
    _progress(f"[GMM selection] representation shape={X.shape}")

    trajectory = fit_trajectory(X)
    verdict = build_verdict(trajectory)

    trajectory.to_csv(TABLES_DIR / "gmm_model_selection_extended.csv", index=False)
    pd.DataFrame([verdict]).to_csv(TABLES_DIR / "gmm_model_selection_verdict.csv", index=False)
    plot_trajectory(trajectory, verdict)

    _progress("\nBIC/AIC trajectory:")
    _progress(trajectory.to_string(index=False))
    _progress("\nVerdict:")
    _progress(str(verdict["justification"]))


if __name__ == "__main__":
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    else:
        fast = FAST_MODE
    main(fast=fast)
