"""
Node: resolve_location
Converts the location text from intent into (lat, lon, canonical_name).
If geocoding fails, sets error fields that route the graph to error_response.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.weather.geocoding import GeocodingError, resolve_location

logger = logging.getLogger(__name__)


def resolve_location_node(state: BotState) -> Dict:
    """
    LangGraph node: resolve location to coordinates.

    - Uses location_text from state (populated by parse_intent or session memory).
    - On success: sets lat, lon, resolved_location.
    - On failure: sets error and error_type for routing to error_response.
    """
    location_text = state.get("location_text")

    # If lat/lon already resolved in a previous turn and location unchanged, reuse.
    existing_lat = state.get("lat")
    existing_lon = state.get("lon")
    existing_location = state.get("resolved_location")

    if (
        location_text
        and existing_lat is not None
        and existing_lon is not None
        and existing_location
        and (
            location_text.strip().lower() in existing_location.lower()
            or existing_location.lower().startswith(location_text.strip().lower())
        )
    ):
        logger.info(
            "LOCATION RESOLUTION (cached): input='%s' -> resolved='%s' (%.4f, %.4f)",
            location_text, existing_location, existing_lat, existing_lon,
        )
        return {}  # No changes needed

    if not location_text:
        logger.warning("No location provided in intent or session memory.")
        return {
            "error": (
                "No location was provided. Please tell me which city or area you're asking about. "
                "For example: 'Is it safe to cycle in Mumbai?'"
            ),
            "error_type": "location_failure",
        }

    logger.info("LOCATION RESOLUTION: resolving '%s' via geocoder", location_text)

    try:
        lat, lon, canonical = resolve_location(location_text)
        logger.info(
            "LOCATION RESOLUTION: input='%s' -> resolved='%s' (%.4f, %.4f)",
            location_text, canonical, lat, lon,
        )
    except GeocodingError as exc:
        logger.error("Geocoding failed: %s", exc)
        return {
            "error": str(exc),
            "error_type": "location_failure",
        }
    except Exception as exc:
        logger.error("Unexpected geocoding error: %s", exc)
        return {
            "error": f"An unexpected error occurred while resolving the location: {exc}",
            "error_type": "location_failure",
        }

    return {
        "lat": lat,
        "lon": lon,
        "resolved_location": canonical,
        "location_text": location_text,
        "error": None,
        "error_type": None,
    }
