"""
Dynamic field requirements helper.
Reads all required_fields from every loaded SOP and returns the union.
The weather fetcher calls this instead of maintaining a hardcoded list.
This means adding a new SOP automatically propagates its required fields.
"""

from __future__ import annotations

from typing import FrozenSet

from app.policy.loader import get_sops

# Open-Meteo variable name → which API section it belongs to
# "current" fields are available in the `current` block.
# "hourly" fields are available in the `hourly` block.
# "daily" fields are available in the `daily` block.
_FIELD_SECTION: dict[str, str] = {
    "temperature_2m": "current",
    "apparent_temperature": "current",
    "relative_humidity_2m": "current",
    "precipitation": "current",
    "precipitation_probability": "hourly",
    "weathercode": "current",
    "visibility": "current",
    "wind_speed_10m": "current",
    "wind_gusts_10m": "current",
    "uv_index": "hourly",
    "precipitation_sum": "daily",       # maps to precipitation_sum_today + next_2d
}

# WeatherFacts field → Open-Meteo API variable name
_FACTS_TO_API: dict[str, str] = {
    "temperature_2m": "temperature_2m",
    "apparent_temperature": "apparent_temperature",
    "relative_humidity_2m": "relative_humidity_2m",
    "precipitation": "precipitation",
    "precipitation_probability": "precipitation_probability",
    "weathercode": "weather_code",
    "visibility": "visibility",
    "wind_speed_10m": "wind_speed_10m",
    "wind_gusts_10m": "wind_gusts_10m",
    "uv_index": "uv_index",
    # SOP-019 uses precipitation_sum_today and precipitation_sum_next_2d
    # both map to the daily "precipitation_sum" variable
    "precipitation_sum_today": "precipitation_sum",
    "precipitation_sum_next_2d": "precipitation_sum",
}


def get_required_api_fields() -> dict[str, list[str]]:
    """
    Return a dict mapping Open-Meteo API section → list of variable names,
    based on the union of required_fields across all loaded SOPs.

    Example:
        {
            "current": ["temperature_2m", "wind_speed_10m", ...],
            "hourly":  ["uv_index", "precipitation_probability"],
            "daily":   ["precipitation_sum"],
        }
    """
    sops = get_sops()
    needed_facts: FrozenSet[str] = frozenset(
        field
        for sop in sops
        for field in sop.required_fields
    )

    sections: dict[str, set[str]] = {"current": set(), "hourly": set(), "daily": set()}

    for fact_field in needed_facts:
        api_var = _FACTS_TO_API.get(fact_field)
        if api_var is None:
            # Field not mapped — skip; document as limitation
            continue
        section = _FIELD_SECTION.get(api_var) or _FIELD_SECTION.get(fact_field)
        if section and api_var:
            sections[section].add(api_var)

    # Always include weather_code in current (needed for SOP-005 override)
    sections["current"].add("weather_code")

    return {sec: sorted(vars_) for sec, vars_ in sections.items() if vars_}
