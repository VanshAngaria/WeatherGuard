from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import requests

from app.weather.field_requirements import get_required_api_fields
from app.weather.models import WeatherFacts

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SEC = 15

_ALL_HOURLY_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "weather_code",
    "visibility",
    "wind_speed_10m",
    "wind_gusts_10m",
    "uv_index",
]


class WeatherFetchError(Exception):
    """Raised when the weather API call fails or returns unusable data."""


def _safe_get(data: dict, *keys, default=None):
    """Safely traverse nested dict/list structures."""
    obj = data
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        elif isinstance(obj, list) and isinstance(key, int):
            try:
                obj = obj[key]
            except IndexError:
                return default
        else:
            return default
    return obj


def _find_target_hour_index(
    times: list[str],
    cur_time: str,
    requested_time: Optional[str],
) -> Tuple[Optional[int], str, int]:
    """
    Resolve target hour index for temporal queries (e.g. 'this evening', 'tomorrow morning').
    Returns (target_index, time_label, current_hour_index).
    If target_index is None, live instantaneous 'current' conditions should be used.
    """
    cur_date = cur_time.split("T")[0] if "T" in cur_time else ""
    all_dates = sorted(list(set(t.split("T")[0] for t in times if "T" in t)))
    tomorrow_date = all_dates[1] if len(all_dates) > 1 else cur_date

    # Find the hourly index that matches current hour (e.g. '2026-09-06T15')
    cur_hour_idx = 0
    cur_prefix = cur_time[:13] if len(cur_time) >= 13 else ""
    for i, t in enumerate(times):
        if cur_prefix and t.startswith(cur_prefix):
            cur_hour_idx = i
            break

    if not requested_time or requested_time.strip().lower() in ("current", "now", "today", "right now", ""):
        return None, "Current Conditions", cur_hour_idx

    req = requested_time.strip().lower()
    is_tomorrow = "tomorrow" in req
    target_date = tomorrow_date if is_tomorrow else cur_date
    day_name = "Tomorrow" if is_tomorrow else "This"

    target_hour = None
    period_name = ""

    if "morning" in req:
        target_hour = "08:00"
        period_name = f"{day_name} Morning"
    elif "afternoon" in req:
        target_hour = "14:00"
        period_name = f"{day_name} Afternoon"
    elif "evening" in req:
        target_hour = "18:00"
        period_name = f"{day_name} Evening"
    elif "night" in req or "tonight" in req:
        target_hour = "21:00"
        period_name = "Tomorrow Night" if is_tomorrow else "Tonight"
    elif is_tomorrow:
        target_hour = "12:00"
        period_name = "Tomorrow Midday"
    else:
        # Check for explicit hour, e.g. '5pm', '6pm', '18:00'
        m = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', req)
        if m:
            h = int(m.group(1))
            ampm = m.group(3)
            if ampm and ampm.lower() == 'pm' and h < 12:
                h += 12
            elif ampm and ampm.lower() == 'am' and h == 12:
                h = 0
            target_hour = f"{h:02d}:00"
            period_name = f"{day_name} {target_hour}"

    if target_hour:
        search_prefix = f"{target_date}T{target_hour[:2]}"
        for i, t in enumerate(times):
            if t.startswith(search_prefix):
                label = f"Forecast Conditions ({period_name} · {target_hour})"
                return i, label, cur_hour_idx

    return None, "Current Conditions", cur_hour_idx


def fetch_weather(lat: float, lon: float, requested_time: Optional[str] = "current") -> WeatherFacts:
    """Fetch live meteorological observations or indexed hourly forecasts from Open-Meteo API."""
    api_fields = get_required_api_fields()

    current_vars = set(api_fields.get("current", []))
    current_vars.update(["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m", "wind_gusts_10m", "visibility"])

    hourly_vars = set(api_fields.get("hourly", []))
    hourly_vars.update(_ALL_HOURLY_FIELDS)

    daily_vars = api_fields.get("daily", ["precipitation_sum"])

    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
        "current": ",".join(sorted(current_vars)),
        "hourly": ",".join(sorted(hourly_vars)),
        "daily": ",".join(daily_vars),
        "forecast_days": 3,
    }

    try:
        response = requests.get(_FORECAST_URL, params=params, timeout=_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise WeatherFetchError(f"Open-Meteo API request failed: {exc}") from exc

    if "current" not in data and "hourly" not in data:
        raise WeatherFetchError("Open-Meteo response missing 'current' and 'hourly' blocks.")

    return _parse_facts(data, requested_time=requested_time)


def _parse_facts(data: dict, requested_time: Optional[str] = "current") -> WeatherFacts:
    """Parse Open-Meteo JSON into structured WeatherFacts, dynamically indexing hourly forecasts if requested."""
    current = data.get("current", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    times = hourly.get("time", [])
    cur_time = current.get("time", "")

    target_idx, time_label, cur_hour_idx = _find_target_hour_index(times, cur_time, requested_time)

    def get_val(field_name: str, api_key: Optional[str] = None) -> Optional[float]:
        key = api_key or field_name
        # If target hourly index selected (e.g. this evening), pull directly from hourly forecast array
        if target_idx is not None and key in hourly:
            arr = hourly.get(key)
            if arr and 0 <= target_idx < len(arr) and arr[target_idx] is not None:
                return float(arr[target_idx])

        # Otherwise pull instantaneous current observation
        if key in current and current[key] is not None:
            return float(current[key])

        # If only available in hourly (like precipitation_probability, uv_index), use cur_hour_idx
        if key in hourly:
            arr = hourly.get(key)
            if arr and 0 <= cur_hour_idx < len(arr) and arr[cur_hour_idx] is not None:
                return float(arr[cur_hour_idx])
            if arr and len(arr) > 0 and arr[0] is not None:
                return float(arr[0])

        return None

    temperature_2m = get_val("temperature_2m")
    apparent_temperature = get_val("apparent_temperature")
    relative_humidity_2m = get_val("relative_humidity_2m")
    precipitation = get_val("precipitation")

    weathercode_val = get_val("weathercode", "weather_code")
    weathercode = int(weathercode_val) if weathercode_val is not None else None

    visibility = get_val("visibility")
    wind_speed_10m = get_val("wind_speed_10m")
    wind_gusts_10m = get_val("wind_gusts_10m")
    precipitation_probability = get_val("precipitation_probability")
    uv_index = get_val("uv_index")

    daily_precip = daily.get("precipitation_sum", [])
    precipitation_sum_today: Optional[float] = None
    precipitation_sum_next_2d: Optional[float] = None

    if daily_precip and len(daily_precip) >= 1:
        v = daily_precip[0]
        precipitation_sum_today = float(v) if v is not None else None

    if daily_precip and len(daily_precip) >= 3:
        vals = [daily_precip[i] for i in [1, 2] if daily_precip[i] is not None]
        precipitation_sum_next_2d = float(sum(vals)) if vals else None

    return WeatherFacts(
        temperature_2m=temperature_2m,
        apparent_temperature=apparent_temperature,
        relative_humidity_2m=relative_humidity_2m,
        precipitation=precipitation,
        precipitation_probability=precipitation_probability,
        precipitation_sum_today=precipitation_sum_today,
        precipitation_sum_next_2d=precipitation_sum_next_2d,
        weathercode=weathercode,
        visibility=visibility,
        wind_speed_10m=wind_speed_10m,
        wind_gusts_10m=wind_gusts_10m,
        uv_index=uv_index,
        time_label=time_label,
    )

