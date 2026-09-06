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


import re

def resolve_location(place_name: str) -> Tuple[float, float, str]:
    """
    Resolve a human-readable place name to (latitude, longitude, resolved_name).
    Employs smart multi-candidate fallback for varied phrasing (e.g. 'city of Mumbai', 'near Delhi').

    Args:
        place_name: Free-text location (e.g., "Bhopal", "New York, USA", "near Delhi").

    Returns:
        Tuple of (latitude, longitude, canonical_name).

    Raises:
        GeocodingError: If the location cannot be resolved.
    """
    clean = place_name.strip().strip("\"'").strip()
    candidates = [clean]

    # Candidate 2: Strip leading noise words / prepositions
    no_prep = re.sub(
        r'^(?:near|around|in|at|of|the|city of|town of|area of|state of|village of|region of)\s+',
        '',
        clean,
        flags=re.IGNORECASE,
    ).strip()
    if no_prep and no_prep not in candidates:
        candidates.append(no_prep)

    # Candidate 3: If comma present (e.g. "Zirakpur, Punjab, India"), try first segment
    if ',' in clean:
        first_seg = clean.split(',')[0].strip()
        if first_seg and first_seg not in candidates:
            candidates.append(first_seg)

    # Candidate 4: Individual significant words
    words = [
        w for w in re.split(r'[\s,]+', clean)
        if len(w) > 2 and w.lower() not in {
            'near', 'around', 'city', 'area', 'town', 'state', 'south', 'north', 'east', 'west', 'central'
        }
    ]
    for w in words:
        if w not in candidates:
            candidates.append(w)

    last_error = None
    for cand in candidates:
        params = {
            "name": cand,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        try:
            response = requests.get(_GEOCODING_URL, params=params, timeout=_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            last_error = exc
            continue

        results = data.get("results")
        if not results:
            continue

        best = results[0]
        lat = best.get("latitude")
        lon = best.get("longitude")
        name = best.get("name", cand)
        country = best.get("country", "")
        admin1 = best.get("admin1", "")

        if lat is None or lon is None:
            continue

        parts = [p for p in [name, admin1, country] if p]
        canonical = ", ".join(parts)

        logger.info("Resolved '%s' (candidate '%s') → %s (%.4f, %.4f)", place_name, cand, canonical, lat, lon)
        return float(lat), float(lon), canonical

    if last_error:
        raise GeocodingError(f"Geocoding API request failed: {last_error}")

    raise GeocodingError(
        f"No location found for '{place_name}'. "
        "Please provide a more specific location name."
    )
