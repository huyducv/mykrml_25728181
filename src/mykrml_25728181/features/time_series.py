"""Leakage-safe supervised feature and target construction for daily time series."""

from __future__ import annotations

import re
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LAG_PATTERN = re.compile(r"^(?P<column>.+)_lag_(?P<days>\d+)d$")
ROLLING_PATTERN = re.compile(
    r"^(?P<column>.+)_past_(?P<days>\d+)d_(?P<aggregation>mean|sum|max)$"
)


def _ordered_daily_frame(
    frame: pd.DataFrame, date_column: str, *, require_continuous: bool = True
) -> pd.DataFrame:
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    result = result.sort_values(date_column).reset_index(drop=True)
    if result[date_column].duplicated().any():
        raise ValueError("Time-series dates must be unique")
    if require_continuous and not result.empty:
        expected = pd.date_range(result[date_column].min(), result[date_column].max(), freq="D")
        if len(expected.difference(pd.DatetimeIndex(result[date_column]))):
            raise ValueError("Time-series data must contain continuous daily dates")
    return result


def add_future_targets(
    frame: pd.DataFrame,
    target_specs: Mapping[str, Tuple[str, int]],
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add future targets by calendar date rather than by row position.

    ``target_specs`` maps each output column to ``(source_column, horizon_days)``.
    Missing calendar dates therefore produce missing targets instead of silently
    shifting values from the wrong day.
    """
    result = _ordered_daily_frame(frame, date_column, require_continuous=False)
    for output_column, (source_column, horizon_days) in target_specs.items():
        if source_column not in result:
            raise ValueError(f"Missing target source column: {source_column}")
        if horizon_days <= 0:
            raise ValueError("Forecast horizons must be positive")
        source_by_date = result.set_index(date_column)[source_column]
        future_dates = result[date_column] + pd.to_timedelta(horizon_days, unit="D")
        result[output_column] = future_dates.map(source_by_date)
    return result


def build_time_series_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Build ordered current, lag, past-only rolling and calendar features.

    Feature names define the operation, for example ``temperature_lag_7d`` or
    ``rain_past_3d_sum``. Rolling features always exclude the current day.
    """
    daily = _ordered_daily_frame(frame, date_column)
    dates = daily[date_column]
    features: dict[str, pd.Series] = {date_column: dates}

    for feature in feature_columns:
        lag_match = LAG_PATTERN.match(feature)
        rolling_match = ROLLING_PATTERN.match(feature)
        if feature in daily.columns:
            values = daily[feature]
        elif lag_match:
            source = lag_match.group("column")
            if source not in daily:
                raise ValueError(f"Unknown source column for {feature}")
            values = daily[source].shift(int(lag_match.group("days")))
        elif rolling_match:
            source = rolling_match.group("column")
            if source not in daily:
                raise ValueError(f"Unknown source column for {feature}")
            window = int(rolling_match.group("days"))
            aggregation = rolling_match.group("aggregation")
            rolling = daily[source].shift(1).rolling(window, min_periods=window)
            values = getattr(rolling, aggregation)()
        elif feature == "month":
            values = dates.dt.month
        elif feature == "day_of_year":
            values = dates.dt.dayofyear
        elif feature == "day_of_year_sin":
            values = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
        elif feature == "day_of_year_cos":
            values = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
        else:
            raise ValueError(f"Unsupported time-series feature: {feature}")
        features[feature] = values

    return pd.DataFrame(features, columns=[date_column, *feature_columns])
