"""
Open-Meteo geocoding API client.
Resolves city/place names to (latitude, longitude).
Never guesses coordinates.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT_SEC = 10


class GeocodingError(Exception):
    """Raised when geocoding fails for any reason."""


def resolve_location(place_name: str) -> Tuple[float, float, str]:
    """
    Resolve a human-readable place name to (latitude, longitude, resolved_name).

    Args:
        place_name: Free-text location (e.g., "Bhopal", "New York, USA").

    Returns:
        Tuple of (latitude, longitude, canonical_name).

    Raises:
        GeocodingError: If the location cannot be resolved.
    """
    params = {
        "name": place_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    try:
        response = requests.get(_GEOCODING_URL, params=params, timeout=_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise GeocodingError(f"Geocoding API request failed: {exc}") from exc

    results = data.get("results")
    if not results:
        raise GeocodingError(
            f"No location found for '{place_name}'. "
            "Please provide a more specific location name."
        )

    best = results[0]
    lat = best.get("latitude")
    lon = best.get("longitude")
    name = best.get("name", place_name)
    country = best.get("country", "")
    admin1 = best.get("admin1", "")

    if lat is None or lon is None:
        raise GeocodingError(
            f"Location '{place_name}' resolved but latitude/longitude missing."
        )

    parts = [p for p in [name, admin1, country] if p]
    canonical = ", ".join(parts)

    logger.info("Resolved '%s' → %s (%.4f, %.4f)", place_name, canonical, lat, lon)
    return float(lat), float(lon), canonical
