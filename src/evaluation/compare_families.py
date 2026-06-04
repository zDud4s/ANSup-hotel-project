"""Cross-family comparison for Task 2."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..preprocessing.feature_config import FAST_MODE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR   = PROJECT_ROOT / "tables"
FIGURES_DIR  = PROJECT_ROOT / "figures"
EXP_CSV      = PROJECT_ROOT / "experiments.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {"MiniBatchKMeans", "iKMeans", "GMM"}
OUT_COLUMNS = [
    "method", "variant", "k_star", "sil_mean", "sil_std", "ch_mean", "ch_std",
    "db_mean", "db_std", "ari_mean", "notes",
]


def _progress(message: str) -> None:
    print(message, flush=True)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["method"].isin(METHODS)].copy()
    for col in ["k", "silhouette", "calinski_harabasz", "davies_bouldin", "ari_vs_seed0"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["method", "variant", "k"])
           .agg(sil_mean=("silhouette", "mean"),
                sil_std=("silhouette", "std"),
                ch_mean=("calinski_harabasz", "mean"),
                ch_std=("calinski_harabasz", "std"),
                db_mean=("davies_bouldin", "mean"),
                db_std=("davies_bouldin", "std"),
                ari_seed0_mean=("ari_vs_seed0", "mean"))
           .reset_index())

    stability = pd.read_csv(TABLES_DIR / "task1_2_stability.csv")
    stability = stability.rename(columns={"mean_pairwise_ari": "ari_mean"})
    gmm_summary = pd.read_csv(TABLES_DIR / "task2_gmm_summary.csv")
    gmm_stability = gmm_summary[["method", "variant", "k", "ari_mean"]]
    stability = pd.concat([stability[["method", "variant", "k", "ari_mean"]],
                           gmm_stability],
                          ignore_index=True)
    agg = agg.merge(stability, on=["method", "variant", "k"], how="left")
    agg["ari_mean"] = agg["ari_mean"].fillna(agg["ari_seed0_mean"])
    return agg.drop(columns=["ari_seed0_mean"])


def _representative_rows(df: pd.DataFrame, agg: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    gmm_selection = pd.read_csv(TABLES_DIR / "task2_gmm_selection.csv")
    gmm_k = dict(zip(gmm_selection["variant"], gmm_selection["k_star"]))

    for (method, variant), sub in df.groupby(["method", "variant"]):
        notes = ""
        if method == "MiniBatchKMeans":
            candidates = agg[(agg["method"] == method) & (agg["variant"] == variant)]
            passing = candidates[candidates["ari_mean"] >= 0.80]
            if passing.empty:
                chosen = candidates.sort_values("sil_mean", ascending=False).iloc[0]
                notes = "no k passed ARI gate; selected max silhouette with caveat"
            else:
                chosen = passing.sort_values("sil_mean", ascending=False).iloc[0]
                notes = "max silhouette among k with ARI >= 0.80"
            k_star = int(chosen["k"])
        elif method == "iKMeans":
            modal = sub["k"].mode()
            k_star = int(modal.iloc[0])
            chosen = agg[(agg["method"] == method)
                         & (agg["variant"] == variant)
                         & (agg["k"] == k_star)].iloc[0]
            notes = "modal auto-determined k"
        else:
            k_star = int(gmm_k[variant])
            chosen = agg[(agg["method"] == method)
                         & (agg["variant"] == variant)
                         & (agg["k"] == k_star)].iloc[0]
            notes = "BIC selection with ARI gate"

        rows.append({
            "method": method,
            "variant": variant,
            "k_star": k_star,
            "sil_mean": chosen["sil_mean"],
            "sil_std": chosen["sil_std"],
            "ch_mean": chosen["ch_mean"],
            "ch_std": chosen["ch_std"],
            "db_mean": chosen["db_mean"],
            "db_std": chosen["db_std"],
            "ari_mean": chosen["ari_mean"],
            "notes": notes,
        })
    return rows


def plot_compare(agg: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    metrics = [
        ("sil_mean", "Silhouette"),
        ("ch_mean", "Calinski-Harabasz"),
        ("db_mean", "Davies-Bouldin"),
    ]
    colors = {"standard": "tab:blue", "robust": "tab:orange"}
    markers = {"MiniBatchKMeans": "o", "GMM": "s", "iKMeans": "^"}

    for ax, (metric, title) in zip(axes, metrics):
        for method in ["MiniBatchKMeans", "GMM"]:
            method_df = agg[agg["method"] == method]
            for variant, sub in method_df.groupby("variant"):
                sub = sub.sort_values("k")
                ax.plot(sub["k"], sub[metric], marker=markers[method],
                        color=colors.get(variant), alpha=0.85,
                        label=f"{method} {variant}")
        ik = agg[agg["method"] == "iKMeans"]
        for variant, sub in ik.groupby("variant"):
            ax.scatter(sub["k"], sub[metric], marker=markers["iKMeans"], s=70,
                       color=colors.get(variant), edgecolor="black",
                       label=f"iKMeans {variant}")
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    fig.suptitle("Task 2 - cross-family comparison", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_compare_families.png", bbox_inches="tight", dpi=130)
    plt.close()


def main(fast: bool = FAST_MODE) -> None:
    _ = fast
    _progress("=== Task 2 - cross-family comparison ===")
    _progress(f"  [compare] reading {EXP_CSV}")
    df = _prepare(pd.read_csv(EXP_CSV))
    _progress("  [compare] aggregating family metrics")
    agg = _aggregate(df)
    _progress("  [compare] selecting representative rows")
    rows = _representative_rows(df, agg)
    out = pd.DataFrame(rows, columns=OUT_COLUMNS).sort_values(["variant", "method"])
    _progress("  [compare] writing task2_compare_families.csv")
    out.to_csv(TABLES_DIR / "task2_compare_families.csv", index=False)
    _progress("  [compare] rendering task2_compare_families.png")
    plot_compare(agg, df)
    _progress("Done.")


if __name__ == "__main__":
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    else:
        fast = FAST_MODE
    main(fast=fast)
