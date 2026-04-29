"""Unified experiment CSV logger for clustering runs."""

from __future__ import annotations

import csv
from pathlib import Path

EXP_HEADER = [
    "task", "method", "variant", "k", "seed",
    "silhouette", "calinski_harabasz", "davies_bouldin", "ari_vs_seed0",
    "bic", "aic", "log_likelihood", "n_iter", "converged", "notes",
]


def write_experiments(rows: list[dict], path: str | Path) -> None:
    """Overwrite the experiment CSV with rows keyed to EXP_HEADER."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXP_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EXP_HEADER})


def append_experiments(rows: list[dict], path: str | Path) -> None:
    """Append experiment rows, writing the header only for a new file."""
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXP_HEADER)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EXP_HEADER})
