"""
Node: fetch_weather
Calls the Open-Meteo forecast API to retrieve live weather data.
On failure, sets error fields that route the graph to error_response.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.weather.open_meteo import WeatherFetchError, fetch_weather

logger = logging.getLogger(__name__)


def fetch_weather_node(state: BotState) -> Dict:
    """
    LangGraph node: fetch live weather data.

    - Reads lat/lon from state (set by resolve_location_node).
    - Calls Open-Meteo forecast API.
    - On success: stores WeatherFacts in state.
    - On failure: sets error and error_type for routing to error_response.
    """
    lat = state.get("lat")
    lon = state.get("lon")

    if lat is None or lon is None:
        return {
            "error": "Coordinates not available — cannot fetch weather.",
            "error_type": "weather_failure",
        }

    logger.info("Fetching weather for (%.4f, %.4f).", lat, lon)

    try:
        facts = fetch_weather(lat, lon)
    except WeatherFetchError as exc:
        logger.error("Weather fetch failed: %s", exc)
        return {
            "error": str(exc),
            "error_type": "weather_failure",
        }
    except Exception as exc:
        logger.error("Unexpected weather error: %s", exc)
        return {
            "error": f"An unexpected error occurred while fetching weather data: {exc}",
            "error_type": "weather_failure",
        }

    return {
        "weather_facts": facts,
        "error": None,
        "error_type": None,
    }
