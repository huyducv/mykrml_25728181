import json

import numpy as np
import pandas as pd
import pytest

from mykrml_25728181.data.open_meteo import (
    aggregate_hourly_to_daily,
    fetch_open_meteo,
    open_meteo_frame,
)
from mykrml_25728181.data.sets import split_time_series_by_periods
from mykrml_25728181.features.time_series import (
    add_future_targets,
    build_time_series_features,
)
from mykrml_25728181.models.performance import (
    print_classifier_scores,
    print_regressor_scores,
)


def hourly_payload():
    return {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "temperature_2m": [20.0, 22.0],
            "precipitation": [None, None],
        },
        "hourly_units": {
            "temperature_2m": "°C",
            "precipitation": "mm",
        },
    }


def test_fetch_open_meteo_can_reuse_a_json_cache(tmp_path):
    cache = tmp_path / "weather.json"
    cache.write_text(json.dumps(hourly_payload()), encoding="utf-8")

    payload, from_cache = fetch_open_meteo(
        -33.86,
        151.20,
        "2025-01-01",
        "2025-01-01",
        ["temperature_2m", "precipitation"],
        cache_path=cache,
    )

    assert from_cache is True
    assert payload == hourly_payload()


def test_open_meteo_frame_validates_lengths_and_units():
    payload = hourly_payload()
    payload["hourly"]["precipitation"] = [0.0, 0.2]
    frame = open_meteo_frame(
        payload,
        ["temperature_2m", "precipitation"],
        expected_units={"temperature_2m": "°C", "precipitation": "mm"},
    )
    assert frame["time"].is_monotonic_increasing

    broken = hourly_payload()
    broken["hourly"]["temperature_2m"] = [20.0]
    with pytest.raises(ValueError, match="inconsistent lengths"):
        open_meteo_frame(broken, ["temperature_2m", "precipitation"])


def test_hourly_aggregation_preserves_all_missing_sums():
    frame = pd.DataFrame(hourly_payload()["hourly"])
    daily = aggregate_hourly_to_daily(
        frame,
        {
            "temperature_2m": ("temperature_2m", "mean"),
            "precipitation": ("precipitation", lambda values: values.sum(min_count=1)),
        },
    )

    assert daily.loc[0, "temperature_2m"] == 21.0
    assert pd.isna(daily.loc[0, "precipitation"])


def test_future_targets_follow_calendar_dates():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "target": [10.0, 20.0, 30.0],
        }
    )
    result = add_future_targets(frame, {"target_t_plus_2": ("target", 2)})

    assert result.loc[0, "target_t_plus_2"] == 30.0
    assert pd.isna(result.loc[1, "target_t_plus_2"])


def test_time_series_features_are_past_only_and_ordered():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=8),
            "temperature": np.arange(1.0, 9.0),
            "rain": np.arange(1.0, 9.0),
        }
    )
    columns = [
        "temperature",
        "temperature_lag_1d",
        "temperature_lag_7d",
        "temperature_past_3d_mean",
        "rain_past_7d_sum",
        "month",
        "day_of_year",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    result = build_time_series_features(frame, columns)
    row = result.iloc[-1]

    assert result.columns.tolist() == ["date", *columns]
    assert row["temperature_lag_1d"] == 7.0
    assert row["temperature_lag_7d"] == 1.0
    assert row["temperature_past_3d_mean"] == 6.0
    assert row["rain_past_7d_sum"] == 28.0


def test_period_split_purges_forecast_boundary_rows():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10),
            "feature": range(10),
            "target": range(10),
        }
    )
    splits, summary = split_time_series_by_periods(
        frame,
        {"train": ("2025-01-01", "2025-01-05"), "test": ("2025-01-06", "2025-01-10")},
        ["feature"],
        ["target"],
        horizon_days=2,
    )

    assert splits["train"]["dates"]["date"].max() == pd.Timestamp("2025-01-03")
    assert summary.set_index("partition").loc["train", "boundary_rows_purged"] == 2


def test_existing_score_functions_now_return_metrics():
    regression = print_regressor_scores([1.0, 3.0], [1.0, 2.0], verbose=False)
    classification = print_classifier_scores([0, 2, 2], [0, 1, 2], verbose=False)

    assert set(regression) == {"rmse", "mae", "r2"}
    assert set(classification) == {
        "accuracy",
        "f1",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
    }
