"""Task 3.3 — StandardScaler vs RobustScaler profile-change comparison.

The robustness plan requires the mandatory StandardScaler-vs-RobustScaler
controlled variant AND a report of how the resulting cluster *profiles* change
across scalers (prof feedback). This module fits the selected family (iK-means)
on the same governed feature matrix under each scaler, profiles each solution
with the shared helpers in ``profile_ikmeans``, and writes a side-by-side
comparison so a reader can see whether the segmentation's meaning survives the
scaler swap.

Outputs:
    tables/task3_scaler_profile_comparison.csv   - per-scaler cluster overview
    report/task3_scaler_profile_comparison.md     - narrative comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing.feature_config import FAST_MODE
from ..preprocessing.pipeline import build_preprocessor
from ..clustering.ikmeans import fit_ikmeans
from .profile_ikmeans import (
    _build_profile_frame,
    _load_frames,
    _overview,
    _top_categories,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCALERS = ("standard", "robust")
# Profile signals reported for the cross-scaler comparison.
KEY_NUMERIC = ["lead_time", "total_nights", "weekend_share", "adr", "is_canceled"]


def _progress(message: str) -> None:
    print(message, flush=True)


def run(fast: bool, seed: int) -> None:
    _progress("=== Task 3.3 - StandardScaler vs RobustScaler profile comparison ===")
    x_input, profiling_frame = _load_frames(fast)

    overviews: dict[str, pd.DataFrame] = {}
    cat_tops: dict[str, pd.DataFrame] = {}
    k_by_scaler: dict[str, int] = {}

    for scaler in SCALERS:
        preproc = build_preprocessor(scaler)
        x = preproc.fit_transform(x_input)
        block_weight = preproc.named_steps["block"].weight_
        labels, _, k_auto = fit_ikmeans(x, seed=seed)
        k_by_scaler[scaler] = k_auto
        frame = _build_profile_frame(x_input, profiling_frame, labels)
        ov = _overview(frame)
        ov.insert(0, "scaler", scaler)
        overviews[scaler] = ov
        cat_tops[scaler] = _top_categories(frame, top_n=3)
        _progress(f"  scaler={scaler}: k={k_auto}, block_weight={block_weight:.3f}, "
                  f"cluster sizes={[int(n) for n in ov['n']]}")

    combined = pd.concat(overviews.values(), ignore_index=True)
    out_csv = TABLES_DIR / "task3_scaler_profile_comparison.csv"
    combined.to_csv(out_csv, index=False)

    # Narrative comparison
    lines = [
        "# StandardScaler vs RobustScaler — profile comparison (iK-means)",
        "",
        "Same governed feature matrix, same algorithm (iK-means), same seed. "
        "Only the numerical scaler changes. The block weight adapts to each "
        "scaler so both blocks still contribute equal variance.",
        "",
        "## Auto-determined k and cluster sizes",
        "",
        "| Scaler | k | Cluster sizes (share) |",
        "|---|---:|---|",
    ]
    for scaler in SCALERS:
        ov = overviews[scaler]
        sizes = ", ".join(f"{int(r.n):,} ({r.share:.1%})" for r in ov.itertuples(index=False))
        lines.append(f"| {scaler} | {k_by_scaler[scaler]} | {sizes} |")

    lines += ["", "## Key numeric signals per cluster", ""]
    cols = [c for c in KEY_NUMERIC]
    header = "| Scaler | Cluster | " + " | ".join(cols) + " |"
    sep = "|---|---:|" + "|".join(["---:"] * len(cols)) + "|"
    lines += [header, sep]
    for scaler in SCALERS:
        ov = overviews[scaler]
        for r in ov.itertuples(index=False):
            vals = []
            for c in cols:
                mean = getattr(r, f"{c}_mean", float("nan"))
                vals.append(f"{mean:.2%}" if c == "is_canceled" else f"{mean:.2f}")
            lines.append(f"| {scaler} | {int(r.cluster)} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Reading",
        "",
        "If the cluster sizes and the direction of the key signals (long vs short "
        "lead time, high vs low cancellation exposure, channel mix) line up across "
        "the two scalers, the segmentation's *meaning* is scaler-robust even when "
        "the exact index values shift. Divergence here would mean a conclusion "
        "depends on the scaling choice and must be reported as such.",
        "",
    ]
    out_md = REPORT_DIR / "task3_scaler_profile_comparison.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    _progress(f"Saved {out_csv}")
    _progress(f"Saved {out_md}")


def main() -> None:
    fast = FAST_MODE
    if "--full" in sys.argv:
        fast = False
    elif "--fast" in sys.argv:
        fast = True
    run(fast=fast, seed=0)


if __name__ == "__main__":
    main()
