from __future__ import annotations

from typing import FrozenSet

from app.policy.loader import get_sops

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
    "precipitation_sum": "daily",
}

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
    "precipitation_sum_today": "precipitation_sum",
    "precipitation_sum_next_2d": "precipitation_sum",
}


def get_required_api_fields() -> dict[str, list[str]]:
    """Determine dynamic API fields required based on loaded SOPs."""
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
            continue
        section = _FIELD_SECTION.get(api_var) or _FIELD_SECTION.get(fact_field)
        if section and api_var:
            sections[section].add(api_var)

    sections["current"].add("weather_code")
    return {sec: sorted(vars_) for sec, vars_ in sections.items() if vars_}
