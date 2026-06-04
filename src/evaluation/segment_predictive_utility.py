"""Post-hoc predictive utility of the headline segments (RQ4 / RQ5).

This module quantifies *how much operationally meaningful, cancellation-relevant
structure* the leakage-free segmentation captures, by asking a question the
clustering itself never saw: given only a booking's headline cluster label, how
well can we anticipate whether it was eventually cancelled?

`is_canceled` is used **strictly post-hoc** as an evaluation target here; it is
never a clustering input (see the leakage controls in feature governance). The
clustering partition is the same headline solution reported in the paper:
iK-means + StandardScaler, k = 7, seed = SEEDS[0].

We compare three discriminators of the held-out cancellation outcome on a fixed
stratified test split:

* **chance** (AUC 0.5) - the no-information baseline;
* **cluster-only** - score each booking by the *training* cancellation rate of
  its cluster (the only information is which of the 7 segments it fell into);
* **full index-time** - Logistic Regression and Random Forest fit on the entire
  preprocessed Euclidean feature matrix (every index-time feature available to
  the clustering).

The headline number is the *fraction of attainable signal recovered by the
cluster label alone*, `(AUC_cluster - 0.5) / (AUC_full - 0.5)`: if seven
segments recover most of what a full-feature model extracts, the segmentation is
capturing the cancellation-relevant behaviour rather than arbitrary partitions.
We additionally report Cramer's V and normalised mutual information between the
cluster label and the outcome, and the Random Forest feature importances - which
independently surface the axes (deposit type, lead time, ADR) along which the
segments separate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..clustering.ikmeans import fit_ikmeans
from ..preprocessing.feature_config import CLUSTER_CATEGORICAL, FAST_MODE, SEEDS
from ..preprocessing.pipeline import build_preprocessor, get_feature_names
from ..utils.experiment_logger import (
    append_experiments,
    build_run_meta,
    to_parameters_json,
)
from .profile_headline import HEADLINE_NUMERICAL, POSTHOC_OUTCOME
from .profile_ikmeans import _load_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _progress(message: str) -> None:
    print(message, flush=True)


def _cramers_v(labels: np.ndarray, outcome: np.ndarray) -> float:
    """Bias-corrected Cramer's V between the cluster label and the outcome."""
    table = pd.crosstab(labels, outcome).to_numpy()
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def _cluster_rate_scores(labels_train, y_train, labels_eval) -> np.ndarray:
    """Score each eval booking by the train cancellation rate of its cluster."""
    rates = pd.Series(y_train).groupby(labels_train).mean()
    global_rate = float(np.mean(y_train))
    return np.array([rates.get(c, global_rate) for c in labels_eval], dtype=float)


def run(fast: bool) -> None:
    _progress("=== Segment predictive utility (post-hoc cancellation) ===")
    x_input, profiling_frame = _load_frames(fast)

    if POSTHOC_OUTCOME not in profiling_frame.columns:
        raise SystemExit(f"Post-hoc outcome '{POSTHOC_OUTCOME}' absent from profiling frame.")
    y = pd.to_numeric(profiling_frame[POSTHOC_OUTCOME], errors="coerce").fillna(0).astype(int).to_numpy()

    preproc = build_preprocessor("standard")
    x = preproc.fit_transform(x_input)
    feature_names = get_feature_names(preproc)
    _progress(f"X shape: {x.shape}; cancellation base rate: {y.mean():.4f}")

    seed = SEEDS[0]
    labels, _, k_auto = fit_ikmeans(x, seed=seed)
    _progress(f"Headline partition: k={k_auto} (seed={seed}, scaler=standard)")

    # Fixed stratified split: cluster-only and full-feature models are scored on
    # the SAME held-out test bookings so the comparison is fair.
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)

    # 1) cluster-only discriminator
    cluster_scores_te = _cluster_rate_scores(labels[idx_tr], y[idx_tr], labels[idx_te])
    auc_cluster = roc_auc_score(y[idx_te], cluster_scores_te)

    # 2) full index-time discriminators (linear + non-linear ceilings)
    logreg = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    logreg.fit(x[idx_tr], y[idx_tr])
    auc_lr = roc_auc_score(y[idx_te], logreg.predict_proba(x[idx_te])[:, 1])

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed
    )
    rf.fit(x[idx_tr], y[idx_tr])
    auc_rf = roc_auc_score(y[idx_te], rf.predict_proba(x[idx_te])[:, 1])

    auc_full = max(auc_lr, auc_rf)
    recovered = (auc_cluster - 0.5) / (auc_full - 0.5) if auc_full > 0.5 else float("nan")

    # association measures (whole data, descriptive)
    cramers_v = _cramers_v(labels, y)
    nmi = float(normalized_mutual_info_score(labels, y))

    _progress(
        f"AUC  chance=0.500  cluster-only={auc_cluster:.4f}  "
        f"LR={auc_lr:.4f}  RF={auc_rf:.4f}  -> recovered={recovered:.1%}"
    )
    _progress(f"Cramer's V={cramers_v:.4f}  NMI={nmi:.4f}")

    # --- per-cluster cancellation table ---
    per_cluster = (
        pd.DataFrame({"cluster": labels, "is_canceled": y})
        .groupby("cluster")["is_canceled"]
        .agg(n="size", cancel_rate="mean")
        .reset_index()
        .sort_values("cluster")
    )
    per_cluster["share_pct"] = per_cluster["n"] / per_cluster["n"].sum() * 100
    table_path = TABLES_DIR / "segment_predictive_utility.csv"
    per_cluster.to_csv(table_path, index=False)
    _progress("Per-cluster cancellation rate:")
    _progress(per_cluster.to_string(index=False))
    _progress(f"Saved table: {table_path}")

    # --- RF feature importances (which index-time axes drive cancellation) ---
    importances = (
        pd.DataFrame({"feature": feature_names, "rf_importance": rf.feature_importances_})
        .sort_values("rf_importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_path = TABLES_DIR / "segment_predictive_utility_rf_importance.csv"
    importances.to_csv(imp_path, index=False)
    _progress("Top index-time predictors of cancellation (RF importance):")
    _progress(importances.head(8).to_string(index=False))
    _progress(f"Saved importances: {imp_path}")

    _write_figure(per_cluster, auc_cluster, auc_lr, auc_rf, importances)

    # --- log one summary row to the experiment CSV ---
    run_meta = build_run_meta(fast=fast, n_rows=len(y), seed=seed)
    notes = (
        f"post-hoc cancellation utility; AUC cluster-only={auc_cluster:.4f}, "
        f"LR={auc_lr:.4f}, RF={auc_rf:.4f}, recovered={recovered:.3f}; "
        f"CramersV={cramers_v:.4f}, NMI={nmi:.4f}"
    )
    params = {
        "auc_chance": 0.5,
        "auc_cluster_only": round(auc_cluster, 4),
        "auc_full_logreg": round(auc_lr, 4),
        "auc_full_randomforest": round(auc_rf, 4),
        "fraction_recovered": round(recovered, 4),
        "cramers_v": round(cramers_v, 4),
        "nmi": round(nmi, 4),
        "test_size": 0.30,
        "outcome": POSTHOC_OUTCOME,
    }
    row = {
        "task": "4",
        "method": "SegmentPredictiveUtility",
        "variant": "standard",
        "k": k_auto,
        "seed": seed,
        "parameters": to_parameters_json(params),
        "notes": notes,
    }
    append_experiments([row], EXPERIMENTS_CSV, run_meta=run_meta)
    _progress(f"Logged summary row to {EXPERIMENTS_CSV}")


def _write_figure(per_cluster, auc_cluster, auc_lr, auc_rf, importances) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # (a) per-cluster cancellation rate
    ax = axes[0]
    ax.bar(per_cluster["cluster"].astype(str), per_cluster["cancel_rate"] * 100, color="#4C72B0")
    ax.axhline(per_cluster["cancel_rate"].mean() * 100, color="grey", ls="--", lw=1, label="mean")
    ax.set_title("(a) Post-hoc cancellation rate by segment")
    ax.set_xlabel("Headline segment")
    ax.set_ylabel("Cancellation rate (%)")
    ax.legend()

    # (b) AUC comparison
    ax = axes[1]
    names = ["chance", "cluster-only", "full (LR)", "full (RF)"]
    vals = [0.5, auc_cluster, auc_lr, auc_rf]
    colors = ["#BBBBBB", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.45, max(vals) + 0.06)
    ax.set_title("(b) Cancellation-AUC: segment label vs full features")
    ax.set_ylabel("ROC AUC (held-out test)")
    ax.tick_params(axis="x", rotation=15)

    # (c) top RF importances
    ax = axes[2]
    top = importances.head(8).iloc[::-1]
    ax.barh(top["feature"], top["rf_importance"], color="#8172B3")
    ax.set_title("(c) Top index-time predictors (RF importance)")
    ax.set_xlabel("Random-forest importance")

    path = FIGURES_DIR / "segment_predictive_utility.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    _progress(f"Saved figure: {path}")


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast)


if __name__ == "__main__":
    main()
