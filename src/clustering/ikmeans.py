"""Intelligent k-means (iK-means) — Anomalous Pattern initialisation.

Reference: Mirkin, B. (2012). *Clustering: A Data Recovery Approach* (2nd ed.).

The algorithm extracts clusters one at a time by repeatedly finding the
point farthest from the data's grand mean and running a 2-cluster
k-means seeded with (grand mean, farthest point). The cluster around
the farthest point is recorded as one "anomalous pattern" and removed.
The procedure repeats on the remaining points until they fall below a
size threshold or a maximum number of clusters is reached. The
collected centres are then used as deterministic initialisation for a
final k-means refinement on the full dataset.

Two consequences relevant to the §3 protocol:

* k is **auto-determined** by the algorithm; we still report results
  inside the same predefined `k ∈ {2..k_max}` range as the k-means
  baseline, capping at `k_max` if the heuristic discovers more.
* The extraction step is deterministic (no randomness), so the seed
  only affects tie-breaking inside the final k-means refinement.

Representation note (metric governance): both the anomalous-pattern
discovery step AND the final k-means refinement run on the SAME full
governed feature matrix (the R-EUCLID matrix) used by every other
standard algorithm. There is **no PCA-projected discovery space** here:
this module never imports or applies PCA. PCA appears only in the
`src/evaluation/visualize_*` diagnostic projections, which are plotting
aids and never feed any clustering. Internal indices are therefore
computed in the same representation and metric as the fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

DEFAULT_K_MAX = 8
DEFAULT_MIN_CLUSTER_SIZE = None  # auto: max(20, 0.5% of n)
DEFAULT_MAX_ITER = 100


def _resolve_min_cluster_size(n: int, override: int | None) -> int:
    if override is not None:
        return override
    return max(20, int(0.005 * n))


def _farthest_from(X: np.ndarray, centre: np.ndarray) -> int:
    diffs = X - centre
    return int(np.argmax((diffs * diffs).sum(axis=1)))


def _two_means_around(X: np.ndarray, centre: np.ndarray, anomaly: np.ndarray,
                      max_iter: int = DEFAULT_MAX_ITER) -> tuple[np.ndarray, np.ndarray]:
    """Run a single 2-means initialised at (centre, anomaly).

    Returns the boolean mask of points assigned to the anomalous cluster
    and the updated anomaly centre.
    """
    c0, c1 = centre.copy(), anomaly.copy()
    for _ in range(max_iter):
        d0 = ((X - c0) ** 2).sum(axis=1)
        d1 = ((X - c1) ** 2).sum(axis=1)
        anomaly_mask = d1 < d0
        if not anomaly_mask.any():
            break
        new_c1 = X[anomaly_mask].mean(axis=0)
        if np.allclose(new_c1, c1):
            break
        c1 = new_c1
    return anomaly_mask, c1


def anomalous_pattern_init(X: np.ndarray,
                           k_max: int = DEFAULT_K_MAX,
                           min_cluster_size: int | None = DEFAULT_MIN_CLUSTER_SIZE,
                           k_min: int = 2,
                           ) -> np.ndarray:
    """Return the anomalous-pattern centres.

    Extracts up to `k_max` centres. The residual size threshold is
    `min_cluster_size` (auto-scaled to max(20, 0.5% of n) when None).
    Always emits at least `k_min` centres so the partition is non-degenerate;
    if the residuals fail the threshold before `k_min` centres are found,
    the next farthest-from-grand-mean candidate is used as the centre
    regardless of its surrounding cluster size.
    """
    n = X.shape[0]
    threshold = _resolve_min_cluster_size(n, min_cluster_size)

    grand_mean = X.mean(axis=0)
    centres: list[np.ndarray] = []
    remaining_idx = np.arange(n)

    while len(centres) < k_max and remaining_idx.size >= threshold:
        X_rem = X[remaining_idx]
        local_mean = X_rem.mean(axis=0) if centres else grand_mean
        far_idx_local = _farthest_from(X_rem, local_mean)
        anomaly = X_rem[far_idx_local]
        mask, anom_centre = _two_means_around(X_rem, local_mean, anomaly)
        if mask.sum() < threshold:
            if len(centres) < k_min:
                # Force-add the anomaly itself to guarantee at least k_min centres.
                centres.append(anomaly)
                remaining_idx = remaining_idx[~mask]
                continue
            break
        centres.append(anom_centre)
        remaining_idx = remaining_idx[~mask]

    if not centres:
        centres.append(grand_mean)
    return np.vstack(centres)


def _merge_tiny_clusters(X: np.ndarray, labels: np.ndarray, centres: np.ndarray,
                         min_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop clusters smaller than min_size; reassign their points to the
    next-nearest surviving centre. Iterates until all clusters are valid
    or only one cluster remains.
    """
    while True:
        sizes = np.bincount(labels, minlength=centres.shape[0])
        small = np.where(sizes < min_size)[0]
        if small.size == 0 or sizes.size - small.size <= 1:
            break
        keep_mask = np.ones(centres.shape[0], dtype=bool)
        keep_mask[small] = False
        survivors = centres[keep_mask]
        # Reassign every point to the nearest surviving centre.
        d = ((X[:, None, :] - survivors[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        # Recompute centres of the surviving clusters from the new assignment.
        new_centres = np.vstack([X[new_labels == j].mean(axis=0)
                                 for j in range(survivors.shape[0])])
        labels, centres = new_labels, new_centres
    # Compact label range to 0..k-1 in case of holes.
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    labels = np.array([remap[v] for v in labels])
    centres = centres[uniq]
    return labels, centres


def fit_ikmeans(X: np.ndarray,
                seed: int = 0,
                k_max: int = DEFAULT_K_MAX,
                min_cluster_size: int | None = DEFAULT_MIN_CLUSTER_SIZE,
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit iK-means on `X`.

    Steps:
      1. Anomalous-pattern initialisation (k auto-determined within
         `[2, k_max]`, with `min_cluster_size` auto-scaled to the dataset).
      2. Final k-means refinement on the full data using those centres.
      3. Drop clusters smaller than the size threshold, reassigning their
         points to the next-nearest surviving centre.

    Returns
    -------
    labels : (n_samples,) int array of cluster assignments.
    centres : (k, n_features) array of final cluster centres.
    k : int, the auto-determined number of clusters after merging.
    """
    threshold = _resolve_min_cluster_size(X.shape[0], min_cluster_size)
    init_centres = anomalous_pattern_init(X, k_max=k_max,
                                          min_cluster_size=threshold)
    k = init_centres.shape[0]
    if k == 1:
        labels = np.zeros(X.shape[0], dtype=int)
        return labels, init_centres, 1

    km = KMeans(n_clusters=k, init=init_centres, n_init=1,
                random_state=seed, max_iter=300)
    labels = km.fit_predict(X)
    centres = km.cluster_centers_
    labels, centres = _merge_tiny_clusters(X, labels, centres, threshold)
    return labels, centres, centres.shape[0]
