"""
Loads and validates the raw hotel bookings CSV.
Checks SHA-256, reports shape/dtypes, and flags missingness.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "hotel_bookings_course_release_v1.csv"

EXPECTED_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"

# Helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksum(path: Path = DATA_CSV) -> None:
    actual = sha256(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum mismatch!\n  expected: {EXPECTED_SHA256}\n  got:      {actual}"
        )
    print(f"SHA-256 OK: {actual}")


def load_raw(path: Path = DATA_CSV) -> pd.DataFrame:
    validate_checksum(path)
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def report_shape(df: pd.DataFrame) -> None:
    print(f"\nShape   : {df.shape}")
    print("Dtypes  :")
    print(df.dtypes.value_counts().to_string())


def report_missingness(df: pd.DataFrame) -> pd.DataFrame:
    missing = (
        df.isnull().sum()
        .rename("n_missing")
        .to_frame()
        .assign(pct=lambda x: (x["n_missing"] / len(df) * 100).round(2))
        .query("n_missing > 0")
        .sort_values("pct", ascending=False)
    )
    if missing.empty:
        print("\nNo missing values.")
    else:
        print("\nMissing values:")
        print(missing.to_string())
    return missing


# Entry point
def main() -> pd.DataFrame:
    if not DATA_CSV.exists():
        print(f"ERROR: data file not found at {DATA_CSV}", file=sys.stderr)
        sys.exit(1)

    df = load_raw()
    report_shape(df)
    report_missingness(df)
    return df


if __name__ == "__main__":
    main()
