"""Internal validity indices and stability measure for the §3 protocol.

All indices are computed in the **same representation and metric space**
as the clustering input (Euclidean over the scaled-numerical / one-hot
matrix). Silhouette is the most expensive (O(n^2) pairwise distances),
so we accept a `sample_size` argument that subsamples for evaluation
while keeping the full assignment intact.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _stratified_silhouette(X: np.ndarray,
                           labels: np.ndarray,
                           sample_size: int,
                           seed: int) -> float:
    """Silhouette on a stratified subsample.

    Used as a fallback when sklearn's random subsampling would drop a
    very small cluster entirely (which raises "Number of labels is 1").
    Each cluster contributes at least `min_per_cluster` points so the
    silhouette is defined and reflects all clusters.
    """
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(labels, return_counts=True)
    n_clusters = len(uniq)
    min_per_cluster = max(20, sample_size // (4 * n_clusters))
    base_idx_parts = []
    for cl, count in zip(uniq, counts):
        cl_idx = np.where(labels == cl)[0]
        n_take = int(min(count, min_per_cluster))
        sel = rng.choice(cl_idx, size=n_take, replace=False)
        base_idx_parts.append(sel)
    base_idx = np.concatenate(base_idx_parts)
    remaining = max(0, sample_size - len(base_idx))
    if remaining > 0:
        leftover = np.setdiff1d(np.arange(len(labels)), base_idx,
                                assume_unique=False)
        n_extra = int(min(remaining, len(leftover)))
        if n_extra > 0:
            extra = rng.choice(leftover, size=n_extra, replace=False)
            sample_idx = np.concatenate([base_idx, extra])
        else:
            sample_idx = base_idx
    else:
        sample_idx = base_idx
    return float(silhouette_score(X[sample_idx], labels[sample_idx],
                                  metric="euclidean"))


def compute_indices(X: np.ndarray,
                    labels: np.ndarray,
                    silhouette_sample_size: int | None = None,
                    seed: int = 0) -> dict:
    """Return Silhouette, Calinski-Harabasz, Davies-Bouldin for `labels`.

    If `silhouette_sample_size` is given and smaller than n, Silhouette is
    estimated on a random subsample (sklearn's `sample_size` argument).
    When a cluster is so small that random subsampling would drop it
    entirely, we fall back to a stratified subsample of the same size.
    A single-cluster solution returns NaNs, since the indices are
    undefined for k=1.
    """
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return {"silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan")}

    try:
        sil = silhouette_score(X, labels, metric="euclidean",
                               sample_size=silhouette_sample_size,
                               random_state=seed)
    except ValueError:
        if silhouette_sample_size is None:
            raise
        sil = _stratified_silhouette(X, labels, silhouette_sample_size, seed)

    return {
        "silhouette": float(sil),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def mean_pairwise_ari(labels_by_seed: dict[int, np.ndarray]) -> float:
    """Stability score: mean ARI across all unordered pairs of seed runs."""
    seeds = sorted(labels_by_seed.keys())
    if len(seeds) < 2:
        return float("nan")
    scores = [
        adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])
        for a, b in combinations(seeds, 2)
    ]
    return float(np.mean(scores))
