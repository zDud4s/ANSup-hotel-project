"""Surrogate decision-tree explanation of the headline segments (RQ4).

The z-scored profile heatmap shows *how* the seven segments differ on average,
but not the crisp decision boundaries that assign a booking to a segment. This
module fits an interpretable, shallow **surrogate decision tree** that predicts
the headline cluster label from the same governed clustering features, turning
the partition into human-readable rules such as

    lead_time > 1.2 (std) AND deposit_type_Non Refund > 0.5  ->  Segment 6.

This is the standard "explain a black-box partition with a transparent model"
technique: the tree is *not* the clusterer, it is a faithful approximation whose
**fidelity** (how often it reproduces the iK-means label) tells us how rule-like
the segmentation is. Cancellation is never involved here - the surrogate is
trained purely on index-time clustering features against the cluster id.

The partition explained is the headline solution: iK-means + StandardScaler,
k = 7, seed = SEEDS[0].

Hyper-parameter governance: the tree depth is a *predefined* constant
(`SURROGATE_MAX_DEPTH`), chosen as the shallowest depth that stays human-readable
while admitting enough leaves to separate seven segments; we report the achieved
fidelity rather than searching depth to maximise it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from ..clustering.ikmeans import fit_ikmeans
from ..preprocessing.feature_config import CLUSTER_CATEGORICAL, CLUSTER_NUMERICAL, FAST_MODE, SEEDS
from ..preprocessing.pipeline import build_preprocessor, get_feature_names
from ..utils.experiment_logger import (
    append_experiments,
    build_run_meta,
    to_parameters_json,
)
from .profile_ikmeans import _load_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Predefined, justified complexity bounds (not searched):
#   depth 4 admits up to 16 leaves - enough to give each of 7 segments a couple
#   of rule paths - while staying small enough to read and plot.
SURROGATE_MAX_DEPTH = 4
SURROGATE_MIN_LEAF_FRAC = 0.01  # a leaf must hold >=1% of bookings to be a "rule"


def _progress(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        # Some Windows consoles are cp1252 and cannot encode symbols like the
        # logical-AND used in plain-language rules; CSV/figure keep them intact.
        enc = sys.stdout.encoding or "ascii"
        print(message.encode(enc, errors="replace").decode(enc), flush=True)


def _humanize_condition(cond: str) -> str:
    """Turn a scaled split such as `hotel_City Hotel <= 0.84` into plain text.

    One-hot dummies are 0/1, so any split separates present from absent; numeric
    features are scaled, so a split just means high vs low on that axis.
    """
    if " <= " in cond:
        feat, _, _thr = cond.partition(" <= ")
        high = False
    else:
        feat, _, _thr = cond.partition(" > ")
        high = True
    feat = feat.strip()

    if feat in CLUSTER_NUMERICAL:
        if feat == "has_kids":
            return "has children" if high else "no children"
        nice = feat.replace("_", " ")
        return f"high {nice}" if high else f"low {nice}"

    # one-hot dummy: strip the "<column>_" prefix to recover the category value
    value = feat
    for col in CLUSTER_CATEGORICAL:
        if feat.startswith(col + "_"):
            value = feat[len(col) + 1:]
            break
    return f"is {value}" if high else f"not {value}"


def _humanize_rule(rule: str) -> str:
    if not rule or rule == "(root)":
        return "(all bookings)"
    return "  ∧  ".join(_humanize_condition(c) for c in rule.split(" AND "))


def _node_question(feature: str) -> tuple[str, str, str]:
    """Plain-language (question, left-branch label, right-branch label) for a split.

    A split sends `<= threshold` left and `> threshold` right. For a 0/1 dummy
    that is absent (left) vs present (right); for a scaled numeric it is low vs
    high on that axis.
    """
    if feature in CLUSTER_NUMERICAL:
        if feature == "has_kids":
            return "children?", "no", "yes"
        return f"{feature.replace('_', ' ')}?", "low", "high"
    value = feature
    for col in CLUSTER_CATEGORICAL:
        if feature.startswith(col + "_"):
            value = feature[len(col) + 1:]
            break
    return f"{value}?", "no", "yes"


def _leaf_rules(tree: DecisionTreeClassifier, feature_names, labels) -> pd.DataFrame:
    """One row per leaf: dominant cluster, size, purity and its decision rule."""
    t = tree.tree_
    feature = t.feature
    threshold = t.threshold
    rows: list[dict] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if feature[node] != -2:  # internal node
            name = feature_names[feature[node]]
            thr = threshold[node]
            recurse(t.children_left[node], conditions + [f"{name} <= {thr:.2f}"])
            recurse(t.children_right[node], conditions + [f"{name} > {thr:.2f}"])
            return
        # tree_.value is normalised to a per-node class distribution in this
        # sklearn version; the leaf size comes from n_node_samples.
        dist = t.value[node][0]
        dominant = int(np.argmax(dist))
        purity = float(dist[dominant] / dist.sum()) if dist.sum() else 0.0
        n_leaf = int(t.n_node_samples[node])
        rows.append(
            {
                "cluster": int(labels[dominant]) if dominant < len(labels) else dominant,
                "n": n_leaf,
                "share_pct": round(n_leaf / int(t.n_node_samples[0]) * 100, 2),
                "purity": round(purity, 3),
                "rule": " AND ".join(conditions) if conditions else "(root)",
            }
        )

    recurse(0, [])
    df = pd.DataFrame(rows)
    return df.sort_values(["cluster", "n"], ascending=[True, False]).reset_index(drop=True)


def run(fast: bool) -> None:
    _progress("=== Surrogate decision-tree explanation of the headline segments ===")
    x_input, _ = _load_frames(fast)

    preproc = build_preprocessor("standard")
    x = preproc.fit_transform(x_input)
    feature_names = get_feature_names(preproc)
    _progress(f"X shape: {x.shape}")

    seed = SEEDS[0]
    labels, _, k_auto = fit_ikmeans(x, seed=seed)
    _progress(f"Headline partition: k={k_auto} (seed={seed}, scaler=standard)")

    min_leaf = max(50, int(SURROGATE_MIN_LEAF_FRAC * len(labels)))
    tree = DecisionTreeClassifier(
        max_depth=SURROGATE_MAX_DEPTH,
        min_samples_leaf=min_leaf,
        random_state=seed,
    )
    tree.fit(x, labels)

    # Fidelity: how faithfully the transparent tree reproduces the iK-means label.
    fit_acc = accuracy_score(labels, tree.predict(x))
    bal_acc = balanced_accuracy_score(labels, tree.predict(x))
    cv = cross_val_score(
        DecisionTreeClassifier(
            max_depth=SURROGATE_MAX_DEPTH, min_samples_leaf=min_leaf, random_state=seed
        ),
        x,
        labels,
        cv=5,
        scoring="accuracy",
    )
    _progress(
        f"Surrogate fidelity: fit-acc={fit_acc:.4f}  balanced-acc={bal_acc:.4f}  "
        f"5-fold cv-acc={cv.mean():.4f}+/-{cv.std():.4f}"
    )

    # --- leaf rules table ---
    rules = _leaf_rules(tree, feature_names, sorted(np.unique(labels)))
    rules["rule_plain"] = rules["rule"].map(_humanize_rule)
    rules_path = TABLES_DIR / "surrogate_tree_rules.csv"
    rules.to_csv(rules_path, index=False)
    _progress("Leaf rules (one decision path per leaf):")
    _progress(rules.to_string(index=False))
    _progress(f"Saved rules: {rules_path}")

    # --- the simplest defining rule per segment (its largest, purest leaf) ---
    signatures = (
        rules.sort_values(["cluster", "purity", "n"], ascending=[True, False, False])
        .groupby("cluster")
        .first()
        .reset_index()[["cluster", "n", "purity", "rule", "rule_plain"]]
    )
    sig_path = TABLES_DIR / "surrogate_tree_signatures.csv"
    signatures.to_csv(sig_path, index=False)
    _progress("Per-segment signature (largest pure leaf):")
    _progress(signatures[["cluster", "n", "purity", "rule_plain"]].to_string(index=False))
    _progress(f"Saved signatures: {sig_path}")

    # --- full text export ---
    text = export_text(tree, feature_names=list(feature_names))
    (TABLES_DIR / "surrogate_tree.txt").write_text(text, encoding="utf-8")

    # --- readable decision-tree diagram with plain-language split questions
    #     (the raw plot_tree is unreadable at depth 4 with 7 classes). ---
    _write_tree_figure(tree, feature_names, k_auto, fit_acc, cv.mean())

    # --- log a summary row ---
    run_meta = build_run_meta(fast=fast, n_rows=len(labels), seed=seed)
    importances = sorted(
        zip(feature_names, tree.feature_importances_), key=lambda t: -t[1]
    )[:5]
    params = {
        "max_depth": SURROGATE_MAX_DEPTH,
        "min_samples_leaf": min_leaf,
        "fit_accuracy": round(float(fit_acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "cv_accuracy_mean": round(float(cv.mean()), 4),
        "top_features": [f for f, _ in importances],
    }
    notes = (
        f"surrogate tree fidelity fit-acc={fit_acc:.3f}, cv-acc={cv.mean():.3f}; "
        f"top split features: {', '.join(f for f, _ in importances)}"
    )
    row = {
        "task": "4",
        "method": "SurrogateTreeExplanation",
        "variant": "standard",
        "k": k_auto,
        "seed": seed,
        "parameters": to_parameters_json(params),
        "notes": notes,
    }
    append_experiments([row], EXPERIMENTS_CSV, run_meta=run_meta)
    _progress(f"Logged summary row to {EXPERIMENTS_CSV}")


def _write_tree_figure(tree, feature_names, k_auto, fit_acc, cv_mean) -> None:
    """Draw the surrogate tree with plain-language questions and yes/no branches.

    sklearn's ``plot_tree`` is unreadable at depth 4 with 7 classes (overlapping
    leaves, scaled thresholds), so we lay the tree out by hand: internal nodes
    ask a plain question, branches are labelled, and leaves are colour-coded by
    the segment they predict.
    """
    t = tree.tree_
    classes = tree.classes_
    LEAF = -1

    xpos: dict[int, float] = {}
    depth: dict[int, int] = {}
    counter = [0]

    def walk(node: int, d: int) -> None:
        depth[node] = d
        if t.children_left[node] == LEAF:
            xpos[node] = float(counter[0])
            counter[0] += 1
        else:
            walk(t.children_left[node], d + 1)
            walk(t.children_right[node], d + 1)
            xpos[node] = (xpos[t.children_left[node]] + xpos[t.children_right[node]]) / 2.0

    walk(0, 0)
    n_leaves = counter[0]
    max_d = max(depth.values())
    cmap = plt.get_cmap("tab10")
    total_n = int(t.n_node_samples[0])

    fig_w = max(15.0, n_leaves * 1.9)
    fig_h = (max_d + 1) * 1.7 + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    def yof(node: int) -> int:
        return -depth[node]

    # edges + branch labels first (so node boxes sit on top)
    for node in xpos:
        if t.children_left[node] == LEAF:
            continue
        _q, left_lab, right_lab = _node_question(feature_names[t.feature[node]])
        for child, lab in ((t.children_left[node], left_lab), (t.children_right[node], right_lab)):
            ax.plot([xpos[node], xpos[child]], [yof(node), yof(child)],
                    color="#B0BEC5", lw=1.2, zorder=1)
            mx, my = (xpos[node] + xpos[child]) / 2, (yof(node) + yof(child)) / 2
            ax.text(mx, my, lab, fontsize=8, ha="center", va="center", color="#37474F",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=2)

    # nodes
    for node in xpos:
        if t.children_left[node] == LEAF:
            dom = int(classes[int(np.argmax(t.value[node][0]))])
            n = int(t.n_node_samples[node])
            share = n / total_n * 100
            purity = float(np.max(t.value[node][0]))
            ax.text(xpos[node], yof(node),
                    f"Seg {dom}\nn={n:,} ({share:.0f}%)\npurity {purity:.2f}",
                    fontsize=8, ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", fc=cmap(dom % 10), ec="black", alpha=0.85))
        else:
            q, _l, _r = _node_question(feature_names[t.feature[node]])
            ax.text(xpos[node], yof(node), q, fontsize=10, ha="center", va="center",
                    fontweight="bold", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ECEFF1", ec="#455A64"))

    ax.set_xlim(-0.8, n_leaves - 0.2)
    ax.set_ylim(-max_d - 0.7, 0.8)
    ax.set_title(
        f"Surrogate decision tree for the headline {k_auto}-segment partition\n"
        f"depth {SURROGATE_MAX_DEPTH}  ·  fidelity: fit-accuracy {fit_acc:.2f}, 5-fold CV {cv_mean:.2f}  "
        f"·  branch = answer to the node's question",
        fontsize=13,
    )
    fig_path = FIGURES_DIR / "surrogate_tree.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _progress(f"Saved figure: {fig_path}")


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast)


if __name__ == "__main__":
    main()
