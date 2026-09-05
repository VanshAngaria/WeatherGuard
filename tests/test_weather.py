"""
Tests for weather module: geocoding and WeatherFacts.
Uses mocked HTTP calls for deterministic testing.
One test uses a real Open-Meteo call (marked with live marker).
"""

from unittest.mock import MagicMock, patch

import pytest

from app.weather.geocoding import GeocodingError, resolve_location
from app.weather.models import WeatherFacts
from app.weather.open_meteo import WeatherFetchError, _parse_facts


# ---------------------------------------------------------------------------
# WeatherFacts model
# ---------------------------------------------------------------------------

def test_weather_facts_all_none():
    """All fields default to None."""
    facts = WeatherFacts()
    d = facts.to_facts_dict()
    assert all(v is None for v in d.values())


def test_weather_facts_to_dict():
    """to_facts_dict returns a flat dict with set values."""
    facts = WeatherFacts(temperature_2m=25.0, weathercode=95)
    d = facts.to_facts_dict()
    assert d["temperature_2m"] == 25.0
    assert d["weathercode"] == 95
    assert d["apparent_temperature"] is None


# ---------------------------------------------------------------------------
# Geocoding (mocked)
# ---------------------------------------------------------------------------

def test_geocoding_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{
            "name": "Mumbai",
            "admin1": "Maharashtra",
            "country": "India",
            "latitude": 19.0760,
            "longitude": 72.8777,
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("app.weather.geocoding.requests.get", return_value=mock_response):
        lat, lon, name = resolve_location("Mumbai")

    assert abs(lat - 19.0760) < 0.001
    assert abs(lon - 72.8777) < 0.001
    assert "Mumbai" in name


def test_geocoding_no_results():
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("app.weather.geocoding.requests.get", return_value=mock_response):
        with pytest.raises(GeocodingError):
            resolve_location("InvalidCity12345")


def test_geocoding_api_error():
    import requests as req
    with patch("app.weather.geocoding.requests.get", side_effect=req.RequestException("timeout")):
        with pytest.raises(GeocodingError):
            resolve_location("Anywhere")


# ---------------------------------------------------------------------------
# Open-Meteo parse_facts (no HTTP call)
# ---------------------------------------------------------------------------

def test_parse_facts_basic():
    """_parse_facts correctly maps current block fields."""
    raw = {
        "current": {
            "temperature_2m": 28.5,
            "apparent_temperature": 32.0,
            "relative_humidity_2m": 72.0,
            "precipitation": 0.0,
            "weather_code": 95,
            "visibility": 800.0,
            "wind_speed_10m": 25.0,
            "wind_gusts_10m": 45.0,
        },
        "hourly": {
            "precipitation_probability": [60.0, 65.0],
            "uv_index": [7.0, 8.0],
        },
        "daily": {
            "precipitation_sum": [70.0, 40.0, 50.0],
        },
    }
    facts = _parse_facts(raw)
    assert facts.temperature_2m == 28.5
    assert facts.weathercode == 95
    assert facts.visibility == 800.0
    assert facts.wind_gusts_10m == 45.0
    assert facts.precipitation_probability == 60.0
    assert facts.uv_index == 7.0
    assert facts.precipitation_sum_today == 70.0
    assert facts.precipitation_sum_next_2d == 90.0  # 40+50


def test_parse_facts_missing_current():
    """
    _parse_facts handles a response with no 'current' block gracefully:
    fields that come from 'current' are None, hourly fields still populate.
    Note: fetch_weather() (not _parse_facts) raises WeatherFetchError when
    BOTH 'current' and 'hourly' are absent.
    """
    raw = {
        "hourly": {
            "precipitation_probability": [30.0],
        },
    }
    # _parse_facts is tolerant — missing current → None current fields
    facts = _parse_facts(raw)
    assert facts.temperature_2m is None
    assert facts.precipitation_probability == 30.0


def test_fetch_weather_raises_on_empty_response():
    """fetch_weather raises WeatherFetchError when response has no current/hourly."""
    import requests as req

    mock_response = MagicMock()
    mock_response.json.return_value = {}  # No current, no hourly
    mock_response.raise_for_status = MagicMock()

    with patch("app.weather.open_meteo.requests.get", return_value=mock_response):
        with pytest.raises(WeatherFetchError):
            from app.weather.open_meteo import fetch_weather
            fetch_weather(51.5, -0.1)


# ---------------------------------------------------------------------------
# Live test (optional — requires network)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_weather_fetch():
    """
    Make a real Open-Meteo API call for London.
    Just checks that WeatherFacts is returned with at least some non-None fields.
    Does NOT check specific values (they change daily).
    """
    from app.weather.geocoding import resolve_location
    from app.weather.open_meteo import fetch_weather

    lat, lon, _ = resolve_location("London")
    facts = fetch_weather(lat, lon)
    non_none = [k for k, v in facts.model_dump().items() if v is not None]
    assert len(non_none) >= 5, f"Expected at least 5 non-None fields, got: {non_none}"
