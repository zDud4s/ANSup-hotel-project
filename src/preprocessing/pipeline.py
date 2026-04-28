"""
Builds and fits the preprocessing pipeline for the hotel bookings dataset.

Feature governance is centralised in feature_config.py. This module turns
the raw dataframe into the matrix used for clustering, and saves the
profiling-only frame separately so it can be joined back for post-hoc
cluster interpretation without ever entering the distance computation.

Pipeline structure (clustering input only):
  Numerical  : median imputation -> StandardScaler
  Categorical: 'Unknown' imputation -> RareCategoryGrouper -> OneHotEncoder

Compact seasonality encoding (Rec 3):
  arrival_date_month is mapped to (arrival_month_sin, arrival_month_cos).
  All four raw arrival_date_* columns are dropped from clustering inputs;
  arrival_date_year/week_number/day_of_month are kept in the profiling
  frame for post-hoc temporal narratives.

Outputs (written to data/processed/):
  X_baseline.npy        - clustering matrix (n_samples x n_features)
  feature_names.txt     - one feature name per line
  pipeline_baseline.pkl - fitted ColumnTransformer
  profiling.parquet     - profiling-only frame (post-hoc interpretation)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.validate import load_raw
from .feature_config import (
    CLUSTER_CATEGORICAL,
    CLUSTER_NUMERICAL,
    COUNTRY_MIN_FREQ,
    FAST_MODE,
    FAST_N,
    FAST_SEED,
    MONTH_TO_NUM,
    PROFILING_ONLY,
    RARE_CATEGORY_MIN_FREQ,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV     = PROJECT_ROOT / "data" / "raw" / "hotel_bookings_course_release_v1.csv"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace categories with fewer than min_freq training occurrences with 'Other'.

    A per-column override map allows a stricter threshold for high-cardinality
    columns such as country (Rec 6).
    """

    def __init__(self, min_freq: int = RARE_CATEGORY_MIN_FREQ, per_column: dict | None = None):
        self.min_freq = min_freq
        self.per_column = per_column or {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.keep_ = []
        for col_name, col in X.items():
            threshold = self.per_column.get(col_name, self.min_freq)
            counts = col.value_counts()
            self.keep_.append(set(counts[counts >= threshold].index))
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i, (col_name, col) in enumerate(X.items()):
            X[col_name] = col.where(col.isin(self.keep_[i]), other="Other")
        return X.values


def add_cyclic_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Replace arrival_date_month with cyclic (sin, cos) features.

    Rec 3: a single behaviourally-justified seasonality encoding instead of
    four overlapping calendar fields. Cyclic encoding ensures December and
    January are close in feature space.
    """
    df = df.copy()
    if "arrival_date_month" in df.columns:
        month_num = df["arrival_date_month"].map(MONTH_TO_NUM)
        df["arrival_month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["arrival_month_cos"] = np.cos(2 * np.pi * month_num / 12)
    return df


def split_clustering_and_profiling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clustering_input, profiling_frame).

    The clustering input contains only the variables that define cluster
    identity. The profiling frame keeps the leakage / post-confirmation /
    profiling-only columns that are still useful for describing what each
    cluster looks like after the model has been fit (Rec 2).
    """
    profiling_cols = [c for c in PROFILING_ONLY if c in df.columns]
    leakage_keep = [c for c in ("is_canceled", "reservation_status", "reservation_status_date")
                    if c in df.columns]
    raw_temporal_keep = [c for c in ("arrival_date_year", "arrival_date_week_number",
                                     "arrival_date_day_of_month", "arrival_date_month")
                         if c in df.columns]
    profiling_frame = df[profiling_cols + leakage_keep + raw_temporal_keep].copy()

    clustering_cols = [c for c in (CLUSTER_NUMERICAL + CLUSTER_CATEGORICAL) if c in df.columns]
    clustering_input = df[clustering_cols].copy()
    return clustering_input, profiling_frame


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # Stricter threshold for country to keep nationality dummies from
        # acting as proxy variables for small subgroups (Rec 6).
        ("grouper", RareCategoryGrouper(per_column={"country": COUNTRY_MIN_FREQ})),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, CLUSTER_NUMERICAL),
        ("cat", cat_pipeline, CLUSTER_CATEGORICAL),
    ])


def run(fast: bool = FAST_MODE) -> tuple[np.ndarray, list[str], ColumnTransformer]:
    """Load raw data, apply governance, fit the pipeline, save outputs."""
    df_raw = load_raw()

    if fast:
        df = df_raw.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        print(f"[FAST MODE] Subsampled to {len(df):,} rows (seed={FAST_SEED})")
    else:
        df = df_raw.copy()
        print(f"[FULL MODE] Using all {len(df):,} rows")

    df = add_cyclic_seasonality(df)
    # Split first so the profiling frame can keep the leakage / post-confirmation
    # columns for post-hoc cluster description; only the clustering input is
    # then governed.
    X_input, profiling_frame = split_clustering_and_profiling(df)
    print(f"Clustering numerical   ({len(CLUSTER_NUMERICAL)}): {CLUSTER_NUMERICAL}")
    print(f"Clustering categorical ({len(CLUSTER_CATEGORICAL)}): {CLUSTER_CATEGORICAL}")
    print(f"Profiling-only         ({profiling_frame.shape[1]}): {list(profiling_frame.columns)}")

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_input)

    cat_names = (
        preprocessor.named_transformers_["cat"]["encoder"]
        .get_feature_names_out(CLUSTER_CATEGORICAL)
    )
    feature_names = CLUSTER_NUMERICAL + list(cat_names)

    assert not np.isnan(X).any(), "NaNs found in transformed matrix"
    print(f"\nTransformed matrix : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Numerical : {len(CLUSTER_NUMERICAL)}")
    print(f"  After OHE : {len(cat_names)}")

    suffix = "_fast" if fast else "_full"
    np.save(PROCESSED / f"X_baseline{suffix}.npy", X)
    (PROCESSED / f"feature_names{suffix}.txt").write_text("\n".join(feature_names))
    with open(PROCESSED / f"pipeline_baseline{suffix}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    profiling_frame.to_csv(PROCESSED / f"profiling{suffix}.csv", index=False)

    print(f"Saved to {PROCESSED}")
    return X, feature_names, preprocessor


if __name__ == "__main__":
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    else:
        fast = FAST_MODE
    run(fast=fast)
