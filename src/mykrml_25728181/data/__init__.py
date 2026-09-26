"""Data loading, cleaning and exploration helpers."""

from mykrml_25728181.data.cleaning import coerce_numeric, drop_duplicate_rows
from mykrml_25728181.data.explore import missing_summary, target_summary
from mykrml_25728181.data.open_meteo import (
    aggregate_hourly_to_daily,
    fetch_open_meteo,
    open_meteo_frame,
)
from mykrml_25728181.data.sets import (
    load_sets,
    pop_target,
    save_sets,
    split_sets_by_time,
    split_sets_random,
    split_time_series_by_periods,
    split_train_val,
    subset_x_y,
)

__all__ = [
    "pop_target",
    "save_sets",
    "load_sets",
    "subset_x_y",
    "split_sets_by_time",
    "split_sets_random",
    "split_train_val",
    "coerce_numeric",
    "drop_duplicate_rows",
    "missing_summary",
    "target_summary",
    "fetch_open_meteo",
    "open_meteo_frame",
    "aggregate_hourly_to_daily",
    "split_time_series_by_periods",
]
