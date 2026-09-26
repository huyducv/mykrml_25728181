"""Small, configurable helpers for tabular Open-Meteo data acquisition."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import httpx
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DateValue = Union[str, date, datetime]
Aggregation = Union[str, Callable[[pd.Series], Any]]


def _iso_date(value: DateValue) -> str:
    return pd.Timestamp(value).date().isoformat()


def fetch_open_meteo(
    latitude: float,
    longitude: float,
    start_date: DateValue,
    end_date: DateValue,
    variables: Sequence[str],
    *,
    section: str = "hourly",
    timezone: str = "UTC",
    model: Optional[str] = None,
    endpoint: str = ARCHIVE_URL,
    extra_params: Optional[Mapping[str, Any]] = None,
    cache_path: Optional[Union[str, Path]] = None,
    refresh: bool = False,
    timeout_seconds: float = 60,
) -> Tuple[dict[str, Any], bool]:
    """Fetch a configurable Open-Meteo response, optionally using a JSON cache.

    Returns the response payload and a flag indicating whether it came from the
    cache. HTTP errors are deliberately allowed to propagate so applications can
    map them to their own retry or status-code behaviour.
    """
    if section not in {"hourly", "daily"}:
        raise ValueError("section must be 'hourly' or 'daily'")
    if not variables:
        raise ValueError("At least one Open-Meteo variable is required")

    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Cached Open-Meteo response must be a JSON object")
        return payload, True

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": _iso_date(start_date),
        "end_date": _iso_date(end_date),
        section: ",".join(variables),
        "timezone": timezone,
    }
    if model is not None:
        params["models"] = model
    if extra_params:
        params.update(extra_params)

    response = httpx.get(
        endpoint,
        params=params,
        timeout=timeout_seconds,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Open-Meteo response must be a JSON object")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, False


def open_meteo_frame(
    payload: Mapping[str, Any],
    variables: Sequence[str],
    *,
    section: str = "hourly",
    expected_units: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Validate an Open-Meteo section and return an ordered DataFrame."""
    if section not in {"hourly", "daily"}:
        raise ValueError("section must be 'hourly' or 'daily'")
    data = payload.get(section)
    if not isinstance(data, Mapping):
        raise ValueError(f"Open-Meteo response is missing {section} data")

    required = ("time", *variables)
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Open-Meteo response is missing variables: {missing}")
    lengths = {name: len(data[name]) for name in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Open-Meteo arrays have inconsistent lengths: {lengths}")
    all_missing = [name for name in variables if all(value is None for value in data[name])]
    if all_missing:
        raise ValueError(f"Open-Meteo variables contain no observations: {all_missing}")

    if expected_units is not None:
        units = payload.get(f"{section}_units")
        if not isinstance(units, Mapping):
            raise ValueError(f"Open-Meteo response is missing {section} units")
        wrong_units = {
            name: units.get(name)
            for name, expected in expected_units.items()
            if units.get(name) != expected
        }
        if wrong_units:
            raise ValueError(f"Unexpected Open-Meteo units: {wrong_units}")

    frame = pd.DataFrame({name: data[name] for name in required})
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.sort_values("time").reset_index(drop=True)
    if frame["time"].duplicated().any():
        raise ValueError("Open-Meteo timestamps must be unique")
    return frame


def aggregate_hourly_to_daily(
    frame: pd.DataFrame,
    aggregations: Mapping[str, Tuple[str, Aggregation]],
    *,
    time_column: str = "time",
    date_column: str = "date",
    require_continuous_dates: bool = True,
) -> pd.DataFrame:
    """Aggregate hourly observations using caller-supplied named aggregations."""
    required = {time_column, *(source for source, _ in aggregations.values())}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing hourly columns: {missing}")

    hourly = frame.copy()
    hourly[time_column] = pd.to_datetime(hourly[time_column], errors="raise")
    if hourly[time_column].duplicated().any():
        raise ValueError("Hourly timestamps must be unique")
    hourly[date_column] = hourly[time_column].dt.normalize()

    named_aggregations = {
        output: pd.NamedAgg(column=source, aggfunc=aggregation)
        for output, (source, aggregation) in aggregations.items()
    }
    daily = hourly.groupby(date_column, as_index=False).agg(**named_aggregations)
    if daily[date_column].duplicated().any() or not daily[date_column].is_monotonic_increasing:
        raise ValueError("Daily dates must be unique and ordered")
    if require_continuous_dates and not daily.empty:
        expected = pd.date_range(daily[date_column].min(), daily[date_column].max(), freq="D")
        if len(expected.difference(pd.DatetimeIndex(daily[date_column]))):
            raise ValueError("Daily data contains missing calendar dates")
    return daily
