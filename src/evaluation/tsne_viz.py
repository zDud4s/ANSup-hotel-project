"""Extension E5: t-SNE visualisation and embedding reliability diagnostics.

t-SNE is used here only as a nonlinear viewing aid for the primary
StandardScaler iK-means solution. No clustering is performed on any t-SNE
embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness

from ..clustering.ikmeans import fit_ikmeans
from ..preprocessing.feature_config import FAST_MODE, FAST_SEED, SEEDS
from ..preprocessing.pipeline import build_preprocessor
from .profile_ikmeans import _load_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_TSNE = 5_000
PERPLEXITIES = (5, 30, 50)
TSNE_SEEDS = (0, 1, 2)
PRIMARY_PERPLEXITY = 30
PRIMARY_SEED = 0
TRUST_N_NEIGHBORS = 10

VIZ_ONLY_STATEMENT = (
    "t-SNE is used for VISUALISATION ONLY; no clustering is performed on the "
    "t-SNE embedding and it is not presented as a clustering result. "
    "Conclusions about clusters come from the primary StandardScaler space."
)
CAVEAT = (
    "Visualisation only. A nonlinear embedding can create or destroy apparent "
    "gaps; trustworthiness quantifies how much local neighbourhood structure "
    "is preserved. Interpret the picture with that caveat."
)


def _progress(message: str) -> None:
    print(message, flush=True)


def _build_standard_space_and_labels(fast: bool) -> tuple[np.ndarray, np.ndarray, int]:
    x_input, _ = _load_frames(fast)
    _progress("Fitting StandardScaler preprocessing")
    preproc = build_preprocessor("standard")
    x = preproc.fit_transform(x_input)
    _progress(f"X shape: {x.shape}")

    _progress(f"Fitting headline iK-means labels with seed={SEEDS[0]}")
    labels, _, k_auto = fit_ikmeans(x, seed=SEEDS[0])
    _progress(f"Headline iK-means k={k_auto}")
    return np.asarray(x, dtype=float), labels, k_auto


def _fixed_subsample(n_rows: int) -> np.ndarray:
    n_sub = min(N_TSNE, n_rows)
    if n_sub == n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(FAST_SEED).choice(n_rows, size=n_sub, replace=False))


def _fit_tsne(x_sub: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return model.fit_transform(x_sub)


def _plot_scatter(ax: plt.Axes, embedding: np.ndarray, labels: np.ndarray, title: str) -> None:
    cmap = plt.get_cmap("tab10")
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.35,
            linewidths=0,
            color=cmap(int(cluster) % 10),
            label=f"cluster {int(cluster)}",
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)


def _write_figures(embeddings: dict[tuple[int, int], np.ndarray],
                   labels_sub: np.ndarray,
                   summary: pd.DataFrame,
                   k_auto: int) -> None:
    main_embedding = embeddings[(PRIMARY_PERPLEXITY, PRIMARY_SEED)]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    _plot_scatter(
        ax,
        main_embedding,
        labels_sub,
        f"t-SNE visualisation of primary-space iK-means k={k_auto} labels",
    )
    ax.legend(title="iK-means label", fontsize=8, title_fontsize=8, loc="best")
    fig.text(0.01, 0.01, "Visualisation only; clusters were fit in StandardScaler space.",
             fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    path = FIGURES_DIR / "e5_tsne_main.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _progress(f"Saved figure: {path}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=False, sharey=False)
    for ax, perplexity in zip(axes, PERPLEXITIES):
        _plot_scatter(
            ax,
            embeddings[(perplexity, PRIMARY_SEED)],
            labels_sub,
            f"perplexity={perplexity}, seed={PRIMARY_SEED}",
        )
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="iK-means label", fontsize=8,
               title_fontsize=8, loc="center right")
    fig.suptitle("t-SNE perplexity sensitivity for the same fixed subsample", fontsize=12)
    plt.tight_layout(rect=(0, 0, 0.91, 0.95))
    path = FIGURES_DIR / "e5_tsne_perplexity_panel.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _progress(f"Saved figure: {path}")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.errorbar(
        summary["perplexity"],
        summary["trust_mean"],
        yerr=summary["trust_std"],
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
    ax.set_title("t-SNE trustworthiness by perplexity", fontsize=11)
    ax.set_xlabel("Perplexity")
    ax.set_ylabel(f"Trustworthiness (n_neighbors={TRUST_N_NEIGHBORS})")
    ax.set_xticks(list(PERPLEXITIES))
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = FIGURES_DIR / "e5_trustworthiness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _progress(f"Saved figure: {path}")


def run(fast: bool) -> None:
    _progress("=== E5 t-SNE visualisation (viz-only) ===")
    _progress(VIZ_ONLY_STATEMENT)
    _progress(CAVEAT)
    _progress(
        f"Fixed subsample rule: t-SNE is ~O(n^2); use n_tsne={N_TSNE:,} rows "
        f"with seed={FAST_SEED} for both --fast and --full. The same indices "
        "are reused for every embedding."
    )
    _progress(
        "Predefined grid: perplexity in {5, 30, 50}, seeds in {0, 1, 2}; "
        "not tuned for prettier plots."
    )

    x, labels, k_auto = _build_standard_space_and_labels(fast)
    sample_idx = _fixed_subsample(len(x))
    x_sub = x[sample_idx]
    labels_sub = labels[sample_idx]
    _progress(f"Using fixed t-SNE subsample: n={len(sample_idx):,}, seed={FAST_SEED}")

    rows: list[dict] = []
    embeddings: dict[tuple[int, int], np.ndarray] = {}
    total = len(PERPLEXITIES) * len(TSNE_SEEDS)
    run_no = 0
    start = perf_counter()
    for perplexity in PERPLEXITIES:
        for seed in TSNE_SEEDS:
            run_no += 1
            _progress(f"  [t-SNE {run_no}/{total}] perplexity={perplexity} seed={seed}")
            embedding = _fit_tsne(x_sub, perplexity=perplexity, seed=seed)
            score = trustworthiness(x_sub, embedding, n_neighbors=TRUST_N_NEIGHBORS)
            embeddings[(perplexity, seed)] = embedding
            rows.append({
                "perplexity": perplexity,
                "seed": seed,
                "trustworthiness": float(score),
            })
            _progress(
                f"  [t-SNE {run_no}/{total}] trustworthiness={score:.4f}"
            )
    elapsed = perf_counter() - start
    _progress(f"Completed 9 t-SNE embeddings in {elapsed / 60:.2f} minutes")

    trust = pd.DataFrame(rows)
    summary = (trust.groupby("perplexity", as_index=False)
               .agg(trust_mean=("trustworthiness", "mean"),
                    trust_std=("trustworthiness", "std")))

    trust_path = TABLES_DIR / "e5_trustworthiness.csv"
    summary_path = TABLES_DIR / "e5_trustworthiness_summary.csv"
    trust.to_csv(trust_path, index=False)
    summary.to_csv(summary_path, index=False)
    _progress(f"Saved table: {trust_path}")
    _progress(f"Saved table: {summary_path}")

    _progress("Trustworthiness summary (mean +/- std over 3 seeds):")
    for row in summary.itertuples(index=False):
        _progress(f"  perplexity={int(row.perplexity):>2}: {row.trust_mean:.4f} +/- {row.trust_std:.4f}")

    _write_figures(embeddings, labels_sub, summary, k_auto)


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast)


if __name__ == "__main__":
    main()
