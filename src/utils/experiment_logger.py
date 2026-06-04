"""Unified experiment CSV logger for clustering runs.

Schema note (Rec: core template columns)
-----------------------------------------
Every experiment row carries provenance columns required by the course
experiment-logging template:

* ``run_id``      - unique id for the script invocation that produced the row.
* ``date``        - ISO-8601 local timestamp of that invocation.
* ``sample_rule`` - exact subsampling rule used (size + seed for the fast
                    sample, or "all rows" for the full run), so a reader can
                    tell preliminary sample results from full-population ones.
* ``parameters``  - JSON blob of the model hyper-parameters for the row.

``run_id``, ``date`` and ``sample_rule`` are run-level: they are passed once
per invocation via ``run_meta`` and stamped onto any row that does not already
set them. ``parameters`` is per-row because methods in the same CSV differ.
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

EXP_HEADER = [
    "run_id", "date", "sample_rule",
    "task", "method", "variant", "k", "seed",
    "silhouette", "calinski_harabasz", "davies_bouldin", "ari_vs_seed0",
    "bic", "aic", "log_likelihood", "n_iter", "converged",
    "parameters", "notes",
]

# Run-level provenance fields filled from ``run_meta`` when a row leaves them blank.
_RUN_LEVEL_FIELDS = ("run_id", "date", "sample_rule")


def new_run_id() -> str:
    """Return a short unique identifier for one script invocation."""
    return uuid.uuid4().hex[:12]


def run_timestamp() -> str:
    """Return the current local time as an ISO-8601 string (second precision)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_sample_rule(fast: bool, n: int, seed: int | None = None) -> str:
    """Describe the subsampling rule for the run.

    Fast mode is an explicit, reproducible sample (size + seed) and is
    labelled as preliminary so its results are never confused with the
    full-population run.
    """
    if fast:
        return f"FAST sample: n={n}, seed={seed} (PRELIMINARY - not the full population)"
    return f"FULL population: all {n} rows"


def to_parameters_json(params: dict) -> str:
    """Serialise a model's hyper-parameters to a compact JSON string."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def build_run_meta(fast: bool, n_rows: int, seed: int | None = None) -> dict:
    """Assemble the run-level provenance dict shared by every row of a run."""
    return {
        "run_id": new_run_id(),
        "date": run_timestamp(),
        "sample_rule": make_sample_rule(fast, n_rows, seed),
    }


def _stamped(rows: list[dict], run_meta: dict | None):
    """Yield rows with run-level provenance filled in where the row left it blank."""
    run_meta = run_meta or {}
    for row in rows:
        out = dict(row)
        for field in _RUN_LEVEL_FIELDS:
            if not out.get(field):
                out[field] = run_meta.get(field, "")
        yield {field: out.get(field, "") for field in EXP_HEADER}


def write_experiments(rows: list[dict], path: str | Path, run_meta: dict | None = None) -> None:
    """Overwrite the experiment CSV with rows keyed to EXP_HEADER."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXP_HEADER)
        writer.writeheader()
        for row in _stamped(rows, run_meta):
            writer.writerow(row)


def append_experiments(rows: list[dict], path: str | Path, run_meta: dict | None = None) -> None:
    """Append experiment rows, writing the header only for a new file."""
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXP_HEADER)
        if write_header:
            writer.writeheader()
        for row in _stamped(rows, run_meta):
            writer.writerow(row)
