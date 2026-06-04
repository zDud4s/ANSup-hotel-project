"""
Build and fit the single preprocessing pipeline for clustering.

Feature governance is centralised in feature_config.py. This module turns
the raw dataframe into the matrix used for clustering, and saves the
profiling-only frame separately so it can be joined back for post-hoc
cluster interpretation without ever entering the distance computation.

Pipeline structure (clustering input only):
  Numerical  : median imputation -> scaler
  Categorical: Unknown imputation -> rare grouping -> one-hot encoding
               -> rare dummy removal   (dummies stay 0/1, NOT standardised)
  Blocks     : categorical block multiplied by a single weight that
               equalises the measured total variance of the two blocks

One-hot dummies are intentionally left as raw 0/1 indicators. Variance-
standardising individual dummies (dividing each by its own std) would
inflate rare categories until a handful of bookings dominate the
Euclidean distance. Instead, a single block weight rescales the whole
categorical block so its total variance matches the numerical block's
(prof feedback on one-hot treatment).

The representation is intentionally singular: engineered stay/party
signals replace the raw correlated stay/party fields, rare one-hot
dummies are removed, and numerical/categorical blocks contribute evenly
to Euclidean distance.

Outputs (written to data/processed/):
  X_baseline.npy        - clustering matrix (n_samples x n_features)
  feature_names.txt     - one feature name per line
  pipeline_baseline.pkl - fitted sklearn Pipeline
  profiling.csv         - profiling-only frame (post-hoc interpretation)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from ..data.validate import load_raw
from .feature_config import (
    CLUSTER_CATEGORICAL,
    CLUSTER_NUMERICAL,
    COUNTRY_MIN_FREQ,
    FAST_MODE,
    FAST_N,
    FAST_SEED,
    MONTH_TO_NUM,
    OHE_MIN_PREVALENCE,
    OHE_VARIANCE_THRESHOLD,
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
    """Inject cyclic (sin, cos) features for month and week-of-year.

    Rec 3: behaviourally-justified seasonality encoding. Cyclic encoding
    ensures December/January (or week 52/week 1) are close in feature
    space rather than 11 (or 51) units apart. Month captures coarse
    seasonality, week captures finer signal (school holidays, peak
    weeks). Source columns stay in the dataframe so the profiling frame
    keeps them; the ColumnTransformer selects only the engineered ones.
    """
    df = df.copy()
    if "arrival_date_month" in df.columns:
        month_num = df["arrival_date_month"].map(MONTH_TO_NUM)
        df["arrival_month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["arrival_month_cos"] = np.cos(2 * np.pi * month_num / 12)
    if "arrival_date_week_number" in df.columns:
        # Period 52 keeps ISO week 53 (rare overflow week) coincident
        # with week 1 of the following year, which is behaviourally
        # correct for seasonality purposes.
        week_num = df["arrival_date_week_number"].astype(float)
        df["arrival_week_sin"] = np.sin(2 * np.pi * week_num / 52)
        df["arrival_week_cos"] = np.cos(2 * np.pi * week_num / 52)
    return df


def add_booking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject engineered stay and party-size features.

    The raw stay/party fields encode one booking across several correlated
    columns. These derived fields keep the behavioural signal while reducing
    redundant axes in the clustering distance.
    """
    df = df.copy()
    weekend = df["stays_in_weekend_nights"].astype(float)
    week = df["stays_in_week_nights"].astype(float)
    adults = df["adults"].astype(float)
    children = df["children"].fillna(0).astype(float)
    babies = df["babies"].astype(float)

    total_nights = weekend + week
    df["total_nights"] = total_nights
    df["party_size"] = adults + children + babies
    df["has_kids"] = ((children + babies) > 0).astype(float)
    safe_total = total_nights.where(total_nights > 0, other=1.0)
    df["weekend_share"] = np.where(total_nights > 0, weekend / safe_total, 0.0)
    return df


def split_clustering_and_profiling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clustering_input, profiling_frame).

    The clustering input contains only the variables that define cluster
    identity. The profiling frame keeps the leakage / post-confirmation /
    profiling-only columns that are still useful for describing what each
    cluster looks like after the model has been fit (Rec 2).
    """
    df = add_booking_features(df)

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


class BlockWeighter(BaseEstimator, TransformerMixin):
    """Scale the whole categorical block so its total variance matches the
    numerical block's.

    The one-hot dummies arrive as raw 0/1 columns (not standardised), so a
    rare dummy contributes only p(1-p) variance and a common one close to
    0.25. We deliberately do NOT divide each dummy by its own std: that would
    blow rare categories up to unit variance and let a handful of bookings
    dominate Euclidean distance. Instead we measure the total variance of
    each block on the fit data and apply a single multiplicative weight

        weight = sqrt( sum_var(numerical block) / sum_var(categorical block) )

    to the entire categorical block. This equalises the variance each block
    contributes to the distance and is scaler-agnostic (it adapts to the
    actual numerical variance under StandardScaler or RobustScaler).
    """

    def __init__(self, n_num: int):
        self.n_num = n_num

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_total = X.shape[1]
        n_cat = max(n_total - self.n_num, 1)
        col_var = X.var(axis=0)
        num_var_total = float(col_var[:self.n_num].sum())
        cat_var_total = float(col_var[self.n_num:].sum())
        if cat_var_total > 0:
            self.weight_ = float(np.sqrt(num_var_total / cat_var_total))
        else:
            self.weight_ = 1.0
        self.n_cat_ = int(n_cat)
        self.num_var_total_ = num_var_total
        self.cat_var_total_ = cat_var_total
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.n_num:] = X[:, self.n_num:] * self.weight_
        return X


def _resolve_scaler(scaler_cls=StandardScaler):
    if isinstance(scaler_cls, str):
        scalers = {"standard": StandardScaler, "robust": RobustScaler}
        try:
            return scalers[scaler_cls]
        except KeyError as exc:
            raise ValueError(f"Unknown scaler: {scaler_cls}") from exc
    return scaler_cls


def build_preprocessor(scaler_cls=StandardScaler) -> Pipeline:
    Scaler = _resolve_scaler(scaler_cls)
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  Scaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # Stricter threshold for country to keep nationality dummies from
        # acting as proxy variables for small subgroups (Rec 6).
        ("grouper", RareCategoryGrouper(per_column={"country": COUNTRY_MIN_FREQ})),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("dropvar", VarianceThreshold(threshold=OHE_VARIANCE_THRESHOLD)),
    ])

    column_transformer = ColumnTransformer([ 
        ("num", num_pipeline, CLUSTER_NUMERICAL),
        ("cat", cat_pipeline, CLUSTER_CATEGORICAL),
    ])
    return Pipeline([
        ("ct", column_transformer),
        ("block", BlockWeighter(n_num=len(CLUSTER_NUMERICAL))),
    ])


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    ct = preprocessor.named_steps["ct"]
    cat_pipe = ct.named_transformers_["cat"]
    encoder = cat_pipe.named_steps["encoder"]
    dropvar = cat_pipe.named_steps["dropvar"]
    cat_names_full = encoder.get_feature_names_out(CLUSTER_CATEGORICAL)
    cat_names_kept = cat_names_full[dropvar.get_support()]
    return CLUSTER_NUMERICAL + list(cat_names_kept)


def run(fast: bool = FAST_MODE) -> tuple[np.ndarray, list[str], Pipeline]:
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

    feature_names = get_feature_names(preprocessor)
    n_cat = len(feature_names) - len(CLUSTER_NUMERICAL)
    block_weight = preprocessor.named_steps["block"].weight_

    assert not np.isnan(X).any(), "NaNs found in transformed matrix"
    print(f"\nTransformed matrix : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Numerical : {len(CLUSTER_NUMERICAL)}")
    print(f"  After OHE : {n_cat} kept (prevalence floor={OHE_MIN_PREVALENCE})")
    print(f"  Categorical block weight : {block_weight:.3f}")

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
