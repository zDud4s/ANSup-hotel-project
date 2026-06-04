"""ADR-inclusion sensitivity study for feature governance.

The headline clustering excludes ADR because price is treated as a downstream
consequence of booking choices rather than as a defining input. This module
tests that modelling choice empirically by fitting the headline no-ADR
representation and an otherwise identical representation with ADR added.

ADR is winsorized before use and post-hoc profiling: values are clipped to the
working dataframe's 0.5th and 99.5th percentiles, and missing values are median
imputed. Without this step, one known erroneous adr ~= 5400 record and adr <= 0
values can dominate the StandardScaled numerical block and make ADR look
spuriously irrelevant or unstable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..clustering.ikmeans import fit_ikmeans
from ..data.validate import load_raw
from ..evaluation.metrics import compute_indices
from ..preprocessing.feature_config import (
    CLUSTER_CATEGORICAL,
    CLUSTER_NUMERICAL,
    COUNTRY_MIN_FREQ,
    FAST_MODE,
    FAST_N,
    FAST_SEED,
    OHE_VARIANCE_THRESHOLD,
    RARE_CATEGORY_MIN_FREQ,
    SEEDS,
)
from ..preprocessing.pipeline import (
    BlockWeighter,
    RareCategoryGrouper,
    add_booking_features,
    add_cyclic_seasonality,
    build_preprocessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLES_DIR / "adr_sensitivity.csv"
SEGMENT_PATH = TABLES_DIR / "adr_per_segment.csv"
FIGURE_PATH = FIGURES_DIR / "adr_sensitivity.png"


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_working_frame(fast: bool) -> pd.DataFrame:
    df_raw = load_raw()
    if "adr" not in df_raw.columns:
        raise RuntimeError(
            "## Escalation\n"
            "adr is absent from the raw columns.\n"
            f"Actual columns: {list(df_raw.columns)}"
        )
    if fast:
        df_raw = df_raw.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
    df = add_cyclic_seasonality(df_raw)
    df = add_booking_features(df)
    return df


def _winsorized_adr(df: pd.DataFrame) -> pd.Series:
    adr = pd.to_numeric(df["adr"], errors="coerce")
    lo = float(adr.quantile(0.005))
    hi = float(adr.quantile(0.995))
    adr_clip = adr.clip(lower=lo, upper=hi)
    median = float(adr_clip.median())
    return adr_clip.fillna(median)


def _build_adr_preprocessor() -> Pipeline:
    numeric_cols = CLUSTER_NUMERICAL + ["adr"]
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("grouper", RareCategoryGrouper(
            min_freq=RARE_CATEGORY_MIN_FREQ,
            per_column={"country": COUNTRY_MIN_FREQ},
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("dropvar", VarianceThreshold(threshold=OHE_VARIANCE_THRESHOLD)),
    ])
    column_transformer = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, CLUSTER_CATEGORICAL),
    ])
    return Pipeline([
        ("ct", column_transformer),
        ("block", BlockWeighter(n_num=len(numeric_cols))),
    ])


def _eta_squared(y: pd.Series, labels: np.ndarray) -> float:
    values = y.to_numpy(dtype=float)
    grand_mean = float(np.mean(values))
    total_ss = float(np.sum((values - grand_mean) ** 2))
    if total_ss <= 0:
        return float("nan")

    between_ss = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between_ss += float(group.size * (np.mean(group) - grand_mean) ** 2)
    return float(between_ss / total_ss)


def _per_segment_adr(adr: pd.Series, labels: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"segment": labels, "adr": adr.to_numpy(dtype=float)})
    out = (
        frame.groupby("segment", as_index=False)
        .agg(n=("adr", "size"), adr_mean=("adr", "mean"), adr_median=("adr", "median"))
        .sort_values("segment")
    )
    out["adr_mean"] = out["adr_mean"].round(4)
    out["adr_median"] = out["adr_median"].round(4)
    return out


def _plot_adr_by_segment(adr: pd.Series, labels: np.ndarray, ari: float, eta_sq: float) -> None:
    segments = sorted(np.unique(labels))
    data = [adr.to_numpy(dtype=float)[labels == segment] for segment in segments]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=[str(s) for s in segments], patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7AA6C2")
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#1B1B1B")
        median.set_linewidth(1.5)

    ax.set_xlabel("Baseline no-ADR segment")
    ax.set_ylabel("Winsorized ADR")
    ax.set_title(f"ADR by price-free segment: ARI(base vs +ADR)={ari:.3f}, eta^2={eta_sq:.3f}")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, bbox_inches="tight", dpi=140)
    plt.close()
    _progress(f"Saved {FIGURE_PATH}")


def _conclusion(ari: float, eta_sq: float) -> str:
    change = "materially changes" if ari < 0.80 else "does not materially change"
    separation = "strong" if eta_sq >= 0.14 else "moderate" if eta_sq >= 0.06 else "limited"
    return (
        f"Adding winsorized ADR {change} the headline segmentation "
        f"(ARI={ari:.3f}); baseline price-free segments show {separation} "
        f"post-hoc ADR separation (eta^2={eta_sq:.3f})."
    )


def run(fast: bool = FAST_MODE) -> None:
    _progress("=== ADR-inclusion sensitivity (RQ2 governance) ===")
    df = _load_working_frame(fast)
    adr_winsor = _winsorized_adr(df)
    df_adr = df.copy()
    df_adr["adr"] = adr_winsor

    _progress(f"Working rows: {len(df_adr):,}")
    _progress("  [preprocess] fitting baseline no-ADR StandardScaler transformer")
    X_base = build_preprocessor(StandardScaler).fit_transform(
        df_adr[CLUSTER_NUMERICAL + CLUSTER_CATEGORICAL]
    )
    _progress("  [preprocess] fitting ADR-augmented StandardScaler transformer")
    X_adr = _build_adr_preprocessor().fit_transform(
        df_adr[CLUSTER_NUMERICAL + ["adr"] + CLUSTER_CATEGORICAL]
    )

    _progress("  [iK-means] fitting baseline no-ADR representation")
    labels_base, _, k_base = fit_ikmeans(X_base, seed=SEEDS[0], k_max=8)
    _progress("  [iK-means] fitting ADR-augmented representation")
    labels_adr, _, k_adr = fit_ikmeans(X_adr, seed=SEEDS[0], k_max=8)

    _progress("  [metrics] computing internal indices and cross-partition ARI")
    idx_base = compute_indices(X_base, labels_base, seed=SEEDS[0])
    idx_adr = compute_indices(X_adr, labels_adr, seed=SEEDS[0])
    ari = float(adjusted_rand_score(labels_base, labels_adr))
    eta_sq = _eta_squared(adr_winsor, labels_base)

    per_segment = _per_segment_adr(adr_winsor, labels_base)
    note = (
        "ADR clipped to working-df 0.5th/99.5th percentiles and median-imputed; "
        "baseline and augmented clusterings use identical rows/order."
    )
    summary = pd.DataFrame([{
        "k_base": int(k_base),
        "k_adr": int(k_adr),
        "ari_base_vs_adr": round(ari, 6),
        "sil_base": idx_base["silhouette"],
        "sil_adr": idx_adr["silhouette"],
        "db_base": idx_base["davies_bouldin"],
        "db_adr": idx_adr["davies_bouldin"],
        "adr_eta_squared_across_base_segments": round(eta_sq, 6),
        "n_rows": int(len(df_adr)),
        "note": note,
    }])

    summary.to_csv(SUMMARY_PATH, index=False)
    per_segment.to_csv(SEGMENT_PATH, index=False)
    _progress(f"Saved {SUMMARY_PATH}")
    _progress(f"Saved {SEGMENT_PATH}")
    _plot_adr_by_segment(adr_winsor, labels_base, ari, eta_sq)

    _progress("\nSummary:")
    _progress(f"  k_base={k_base}, k_adr={k_adr}")
    _progress(
        f"  ARI(base vs +ADR)={ari:.3f}; "
        f"baseline ADR eta^2={eta_sq:.3f}"
    )
    _progress("  Baseline per-segment ADR spread:")
    _progress(per_segment.to_string(index=False))
    _progress(f"  Conclusion: {_conclusion(ari, eta_sq)}")
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
