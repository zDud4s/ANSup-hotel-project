"""
Single source of truth for feature governance.

Index time = the moment the booking is confirmed. Every clustering input
must be available at or before that moment. Anything that is updated,
revised, or determined after confirmation belongs in the profiling layer
(post-hoc interpretation only) or is dropped entirely.

Feature roles
-------------
1. LEAKAGE_COLS         dropped, post-event variables (cancellation outcome).
2. POST_CONFIRMATION_COLS dropped, only resolved after the booking is taken.
3. ID_COLS              dropped, identifier-like / no behavioural signal.
4. PROFILING_ONLY       kept in the dataframe but never enter clustering;
                        used only for post-hoc cluster profiling.
5. CLUSTER_NUMERICAL    numerical inputs to the clustering pipeline.
6. CLUSTER_CATEGORICAL  categorical inputs to the clustering pipeline.

Cyclic seasonality encoding
------------------------------------
The four raw arrival_date_* columns carry overlapping but not redundant
information. We follow the course's column_roles.csv recommendation
("consider cyclic/ordinal encoding choices") and encode the two that
carry behavioural signal:

  arrival_date_month        -> (arrival_month_sin, arrival_month_cos)
  arrival_date_week_number  -> (arrival_week_sin,  arrival_week_cos)

Cyclic encoding ensures December and January (or week 52 and week 1) are
close in feature space rather than 11 / 51 units apart. Month captures
high-level seasonality (peak summer, off-season), week captures finer
seasonality (school holidays, Easter, specific peak weeks) that month
alone smooths over.

The remaining two raw temporal fields are dropped:

  arrival_date_year          - artefact of the dataset window (Jul-2015
                               to Aug-2017, only 3 distinct values);
                               behavioural conclusions should be
                               year-invariant.
  arrival_date_day_of_month  - sub-monthly noise, no behavioural meaning
                               at the booking-segmentation timescale;
                               weekday/weekend rhythm is already captured
                               by the stays_in_weekend_nights /
                               stays_in_week_nights split.

All four raw arrival_date_* columns are kept in the profiling frame so
post-hoc cluster narratives can still reference the calendar directly.
"""

# Dropped, post-event outcome variables
LEAKAGE_COLS = [
    "is_canceled",              # booking outcome, unknown at index time
    "reservation_status",       # post-arrival operational state
    "reservation_status_date",  # date of that post-arrival state change
]

# Dropped, only resolved after the booking is confirmed
POST_CONFIRMATION_COLS = [
    "assigned_room_type",   # set on check-in/operational rebalancing, not at booking
    "booking_changes",      # accrued via post-booking modifications
    "days_in_waiting_list", # accrued between booking and confirmation/cancellation
]

# Dropped, identifier-like columns with no behavioural meaning
ID_COLS = [
    "agent",    # travel-agent ID; 13.7% missing
    "company",  # company ID; 94.3% missing
]

# Raw temporal fields handled by the cyclic / drop policy in the docstring
# above. All four are kept in the profiling frame for post-hoc narratives.
RAW_TEMPORAL_COLS = [
    "arrival_date_year",          # dropped: 3-value dataset-window artefact
    "arrival_date_month",         # consumed by cyclic (sin, cos), then dropped
    "arrival_date_week_number",   # consumed by cyclic (sin, cos), then dropped
    "arrival_date_day_of_month",  # dropped: sub-monthly noise
]

# Profiling-only variables: kept for post-hoc cluster profiling,
#    never used as clustering inputs. we kept meal, required_car_parking_spaces, 
#    and total_of_special_requests as profiling variables. ADR is also reserved for profiling because
#    price is a downstream consequence of the booking choices (room type,
#    dates, channel, party size) we cluster on; mixing it in conflates
#    cluster definition with cluster description.
PROFILING_ONLY = [
    "meal",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "adr",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "country",
]

# Numerical clustering inputs.
# Raw stay/party fields are collapsed into behaviourally meaningful
# booking-shape signals in preprocessing.pipeline.add_booking_features.
# All are available at index time and interpretable.
CLUSTER_NUMERICAL = [
    "lead_time",
    "total_nights",
    "party_size",
    "has_kids",
    "weekend_share",
    "arrival_month_sin",  # cyclic month-of-year seasonality
    "arrival_month_cos",
    "arrival_week_sin",   # cyclic week-of-year seasonality (finer grain)
    "arrival_week_cos",
]

# Categorical clustering inputs.
# country is kept with strict interpretive caution: rare-category
# grouping reduces dimensionality, but the variable must be read as
# market-of-origin behaviour, never as nationality, and never as a
# substantive explanation for cluster identity. The risk of proxy
# discrimination must remain explicit in the report and conclusions.
CLUSTER_CATEGORICAL = [
    "hotel",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

# Convenience aggregate, used by the pipeline to drop columns up-front.
DROP_COLS = LEAKAGE_COLS + POST_CONFIRMATION_COLS + ID_COLS

# Rare-category threshold: categories with fewer than this many training
# occurrences are grouped into 'Other' before one-hot encoding.
RARE_CATEGORY_MIN_FREQ = 50

# Country gets a stricter threshold because it is high-cardinality (~93
# unique values) and we explicitly want to avoid sparse nationality dummies
# that would only act as proxies for small subgroups.
COUNTRY_MIN_FREQ = 100

# One-hot prevalence floor. Dummies below this frequency add sparse
# distance dimensions without stable cluster structure, so they are
# dropped after encoding.
OHE_MIN_PREVALENCE = 0.005
OHE_VARIANCE_THRESHOLD = OHE_MIN_PREVALENCE * (1 - OHE_MIN_PREVALENCE)

# Fast-mode subsampling
FAST_MODE = True
FAST_N    = 5_000
FAST_SEED = 42

# Fixed seeds, used uniformly across every clustering experiment.
# Ten seeds satisfy the Milestone-2 minimum (>= 10 runs where randomness
# applies) and give 45 pairs for ARI stability per (method, variant, k).
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Month name -> ordinal, used by the cyclic seasonality transform.
MONTH_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
