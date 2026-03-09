"""
Single source of truth for column inclusion/exclusion decisions.
"""

LEAKAGE_COLS = [
    "is_canceled",              # outcome — not known at booking time
    "reservation_status",       # post-arrival status
    "reservation_status_date",  # date of that status change
]

EXCLUDE_COLS = [
    "agent",    # travel agent numeric ID; 13% missing; not a behaviour feature
    "company",  # company numeric ID; 94% missing; not a behaviour feature
]

DROP_COLS = LEAKAGE_COLS + EXCLUDE_COLS

# Rare-category threshold
# Categories with fewer than MIN_FREQ training occurrences are grouped into
# 'Other' before one-hot encoding, to avoid near-zero dummy columns.
RARE_CATEGORY_MIN_FREQ = 50

# Fast-mode subsampling
FAST_MODE = True
FAST_N    = 5_000
FAST_SEED = 42

SEEDS = [0, 1, 2, 3, 4]  # fixed seeds for reproducibility across experiments
