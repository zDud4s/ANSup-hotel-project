"""Profile the §3 selection-rule winner: iK-means + RobustScaler, k=2 (full mode)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.data.validate import load_raw
from src.preprocessing.feature_config import CLUSTER_CATEGORICAL, CLUSTER_NUMERICAL
from src.preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)
from src.clustering.ikmeans import fit_ikmeans

df = load_raw()
print(f"Loaded {len(df):,} rows")
df = add_cyclic_seasonality(df)
X_input, profiling = split_clustering_and_profiling(df)

preproc = build_preprocessor("robust")
X = preproc.fit_transform(X_input)
print(f"X shape: {X.shape}")

labels, _, k = fit_ikmeans(X, seed=0, k_max=8)
print(f"k={k}, sizes={np.bincount(labels).tolist()}")

# Cluster sizes
sizes = pd.Series(labels).value_counts().sort_index()
print("\n=== Cluster sizes ===")
print(sizes.to_frame("n").assign(pct=lambda d: (d["n"]/len(labels)*100).round(1)))

# Cluster profile from clustering inputs (numerical means + top categorical share)
prof_num = X_input[CLUSTER_NUMERICAL].assign(cluster=labels).groupby("cluster").mean().round(2)
print("\n=== Numerical clustering inputs (mean by cluster) ===")
print(prof_num.T)

# Categorical: top category share per cluster for the most informative cats
print("\n=== Categorical clustering inputs (mode + share) ===")
for cat in CLUSTER_CATEGORICAL:
    crosstab = pd.crosstab(labels, X_input[cat], normalize="index").round(3)
    # show top 3 categories overall per cluster
    print(f"\n  -- {cat} --")
    print(crosstab.T.head(8))

# Profiling-only (post-hoc, not used for clustering)
prof = profiling.assign(cluster=labels)
print("\n=== Profiling-only variables (mean / share by cluster) ===")
num_cols = ["adr", "required_car_parking_spaces", "total_of_special_requests"]
print(prof.groupby("cluster")[num_cols].mean().round(3).T)

print("\n  -- meal share by cluster --")
print(pd.crosstab(prof["cluster"], prof["meal"], normalize="index").round(3).T)

print("\n  -- is_canceled rate by cluster (POST-HOC ONLY) --")
print(prof.groupby("cluster")["is_canceled"].agg(["mean", "count"]).round(3))

print("\n  -- arrival_date_year share by cluster --")
print(pd.crosstab(prof["cluster"], prof["arrival_date_year"], normalize="index").round(3).T)
