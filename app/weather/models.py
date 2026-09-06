"""
WeatherFacts data model.
Single source of truth for all weather numbers used by the policy engine.
All values come from validated Open-Meteo API responses — never from the LLM.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WeatherFacts(BaseModel):
    """
    Typed container for weather values extracted from Open-Meteo.

    Fields:
        temperature_2m:              Air temperature at 2m height (°C)
        apparent_temperature:        Feels-like temperature (°C)
        relative_humidity_2m:        Relative humidity at 2m (%)
        precipitation:               Current precipitation (mm)
        precipitation_probability:   Probability of precipitation (%)
        weathercode:                 WMO weather code (integer)
        visibility:                  Horizontal visibility (meters)
        wind_speed_10m:              Wind speed at 10m (km/h)
        wind_gusts_10m:              Wind gusts at 10m (km/h)
        uv_index:                    UV index (dimensionless)
        precipitation_sum_today:     Daily total precipitation — today (mm)
        precipitation_sum_next_2d:   Sum of precipitation for next 2 days (mm)

    A None value means the field was not available in the API response.
    A missing fact MUST NOT cause any SOP condition to pass.
    """

    temperature_2m: Optional[float] = None
    apparent_temperature: Optional[float] = None
    relative_humidity_2m: Optional[float] = None
    precipitation: Optional[float] = None
    precipitation_probability: Optional[float] = None
    weathercode: Optional[int] = None
    visibility: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    wind_gusts_10m: Optional[float] = None
    uv_index: Optional[float] = None
    precipitation_sum_today: Optional[float] = None
    precipitation_sum_next_2d: Optional[float] = None
    time_label: Optional[str] = "Current Conditions"

    def to_facts_dict(self) -> dict:
        """Return a flat dict suitable for the policy evaluator."""
        return {k: v for k, v in self.model_dump().items() if k != "time_label"}
