"""
Builds and fits the preprocessing pipeline for the hotel bookings dataset.

Pipeline structure:
  Numerical  : median imputation → StandardScaler
  Categorical: 'Unknown' imputation → RareCategoryGrouper → OneHotEncoder

Outputs (written to data/processed/):
  X_baseline.npy        — transformed matrix (n_samples × 82)
  feature_names.txt     — one feature name per line
  pipeline_baseline.pkl — fitted ColumnTransformer (for reuse on new data)
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
    DROP_COLS,
    FAST_MODE,
    FAST_N,
    FAST_SEED,
    RARE_CATEGORY_MIN_FREQ,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV     = PROJECT_ROOT / "data" / "raw" / "hotel_bookings_course_release_v1.csv"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Custom transformer
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace categories with fewer than min_freq training occurrences with 'Other'."""

    def __init__(self, min_freq: int = RARE_CATEGORY_MIN_FREQ):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = [
            set(col.value_counts()[col.value_counts() >= self.min_freq].index)
            for _, col in X.items()
        ]
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i, (col_name, col) in enumerate(X.items()):
            X[col_name] = col.where(col.isin(self.keep_[i]), other="Other")
        return X.values

# Pipeline builder
def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("grouper", RareCategoryGrouper()),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ])


def run(fast: bool = FAST_MODE) -> tuple[np.ndarray, list[str], ColumnTransformer]:
    """
    Load raw data, apply exclusions, fit the pipeline, save outputs.
    Returns (X, feature_names, fitted_preprocessor).
    """
    df_raw = load_raw()

    if fast:
        df = df_raw.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        print(f"[FAST MODE] Subsampled to {len(df):,} rows (seed={FAST_SEED})")
    else:
        df = df_raw.copy()
        print(f"[FULL MODE] Using all {len(df):,} rows")

    df_input = df.drop(columns=DROP_COLS)

    numerical_features   = df_input.select_dtypes(include="number").columns.tolist()
    categorical_features = df_input.select_dtypes(include="str").columns.tolist()

    print(f"Numerical features   ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X = preprocessor.fit_transform(df_input)

    cat_names = (
        preprocessor.named_transformers_["cat"]["encoder"]
        .get_feature_names_out(categorical_features)
    )
    feature_names = numerical_features + list(cat_names)

    assert not np.isnan(X).any(), "NaNs found in transformed matrix"
    print(f"\nTransformed matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Numerical : {len(numerical_features)}")
    print(f"  After OHE : {len(cat_names)}")

    # Save outputs
    suffix = "_fast" if fast else "_full"
    np.save(PROCESSED / f"X_baseline{suffix}.npy", X)
    (PROCESSED / f"feature_names{suffix}.txt").write_text("\n".join(feature_names))
    with open(PROCESSED / f"pipeline_baseline{suffix}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"Saved to {PROCESSED}")
    return X, feature_names, preprocessor


if __name__ == "__main__":
    fast = "--fast" in sys.argv or FAST_MODE
    run(fast=fast)
