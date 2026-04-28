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

Compact seasonality encoding
------------------------------------
The four raw arrival_date_* columns are redundant: arrival_date_year is an
artefact of the dataset window (Jul-2015 to Aug-2017), week_number and
day_of_month overlap with month, and using them all inflates the implicit
distance between bookings that differ on calendar bookkeeping rather than
booking behaviour. We replace them with two cyclic features
(arrival_month_sin, arrival_month_cos) computed from arrival_date_month,
which keeps the seasonal signal compact and behaviourally justified
(December and January are close in calendar space rather than 11 apart).
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

# Dropped, redundant raw temporal fields (replaced by cyclic encoding)
RAW_TEMPORAL_COLS = [
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "arrival_date_month",  # consumed by the cyclic transform, then dropped
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
]

# Numerical clustering inputs.
# All available at index time, all behaviourally interpretable.
CLUSTER_NUMERICAL = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "arrival_month_sin",  # cyclic seasonality
    "arrival_month_cos",  # cyclic seasonality
]

# Categorical clustering inputs.
# country is kept with strict interpretive caution: rare-category
# grouping reduces dimensionality, but the variable must be read as
# market-of-origin behaviour, never as nationality, and never as a
# substantive explanation for cluster identity. The risk of proxy
# discrimination must remain explicit in the report and conclusions.
CLUSTER_CATEGORICAL = [
    "hotel",
    "country",  # interpret with caution; document proxy-discrimination risk
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

# Fast-mode subsampling
FAST_MODE = True
FAST_N    = 5_000
FAST_SEED = 42

# Fixed seeds, used uniformly across every clustering experiment.
SEEDS = [0, 1, 2, 3, 4]

# Month name -> ordinal, used by the cyclic seasonality transform.
MONTH_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
