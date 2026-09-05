"""
Open-Meteo forecast API client.
Fetches live weather data and converts the response to a WeatherFacts object.

Design principles:
- Required fields are read dynamically from all loaded SOPs (field_requirements.py).
- Adding a new SOP with new required_fields propagates automatically.
- Weather numbers flow into WeatherFacts; the LLM never sees raw API JSON.
- Missing fields remain None in WeatherFacts — they do not invent values.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from app.weather.field_requirements import get_required_api_fields
from app.weather.models import WeatherFacts

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SEC = 15


class WeatherFetchError(Exception):
    """Raised when the weather API call fails or returns unusable data."""


def _safe_get(data: dict, *keys, default=None):
    """Safely traverse nested dict/list structure."""
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


def fetch_weather(lat: float, lon: float) -> WeatherFacts:
    """
    Fetch current weather conditions from Open-Meteo.

    Args:
        lat: Latitude (from geocoding).
        lon: Longitude (from geocoding).

    Returns:
        WeatherFacts populated from the API response.

    Raises:
        WeatherFetchError: On API error or unusable response.
    """
    api_fields = get_required_api_fields()

    current_vars = api_fields.get("current", [])
    hourly_vars = api_fields.get("hourly", [])
    daily_vars = api_fields.get("daily", [])

    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }

    if current_vars:
        params["current"] = ",".join(current_vars)
    if hourly_vars:
        params["hourly"] = ",".join(hourly_vars)
    if daily_vars:
        params["daily"] = ",".join(daily_vars)
        # Fetch 3 days to compute today + next 2 days
        params["forecast_days"] = 3

    logger.debug("Open-Meteo request params: %s", params)

    try:
        response = requests.get(_FORECAST_URL, params=params, timeout=_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise WeatherFetchError(f"Open-Meteo API request failed: {exc}") from exc

    if "current" not in data and "hourly" not in data:
        raise WeatherFetchError("Open-Meteo response missing 'current' and 'hourly' blocks.")

    return _parse_facts(data)


def _parse_facts(data: dict) -> WeatherFacts:
    """
    Parse the raw Open-Meteo JSON response into a WeatherFacts object.
    All conversions are explicit; no values are invented.
    """
    current = data.get("current", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    # --- Current block fields ---
    def cur(field: str) -> Optional[float]:
        val = current.get(field)
        return float(val) if val is not None else None

    temperature_2m = cur("temperature_2m")
    apparent_temperature = cur("apparent_temperature")
    relative_humidity_2m = cur("relative_humidity_2m")
    precipitation = cur("precipitation")
    weathercode_raw = current.get("weather_code")
    weathercode = int(weathercode_raw) if weathercode_raw is not None else None
    visibility = cur("visibility")
    wind_speed_10m = cur("wind_speed_10m")
    wind_gusts_10m = cur("wind_gusts_10m")

    # --- Hourly block: take the first available index (nearest hour) ---
    def hourly_first(field: str) -> Optional[float]:
        vals = hourly.get(field)
        if vals and len(vals) > 0 and vals[0] is not None:
            return float(vals[0])
        # Try index 1 as fallback
        if vals and len(vals) > 1 and vals[1] is not None:
            return float(vals[1])
        return None

    precipitation_probability = hourly_first("precipitation_probability")
    uv_index = hourly_first("uv_index")

    # --- Daily block: precipitation sums ---
    daily_precip = daily.get("precipitation_sum", [])
    precipitation_sum_today: Optional[float] = None
    precipitation_sum_next_2d: Optional[float] = None

    if daily_precip and len(daily_precip) >= 1:
        v = daily_precip[0]
        precipitation_sum_today = float(v) if v is not None else None

    if daily_precip and len(daily_precip) >= 3:
        vals = [daily_precip[i] for i in [1, 2] if daily_precip[i] is not None]
        precipitation_sum_next_2d = float(sum(vals)) if vals else None

    facts = WeatherFacts(
        temperature_2m=temperature_2m,
        apparent_temperature=apparent_temperature,
        relative_humidity_2m=relative_humidity_2m,
        precipitation=precipitation,
        precipitation_probability=precipitation_probability,
        weathercode=weathercode,
        visibility=visibility,
        wind_speed_10m=wind_speed_10m,
        wind_gusts_10m=wind_gusts_10m,
        uv_index=uv_index,
        precipitation_sum_today=precipitation_sum_today,
        precipitation_sum_next_2d=precipitation_sum_next_2d,
    )

    logger.info("WeatherFacts extracted: %s", facts.model_dump(exclude_none=True))
    return facts
