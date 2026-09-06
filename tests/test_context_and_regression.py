"""
Comprehensive test suite for conversational context resolution and regression scenarios.
Covers:
1. Explicit location + activity
2. Time-only follow-up
3. Location-only follow-up
4. Activity-only follow-up
5. Date + time follow-up
6. Context replacement (activity & location replacement)
7. New session with incomplete context (clarification routing)
8. Gemini failure + heuristic fallback
9. Weather API failure
10. No applicable SOP
"""

import uuid
from unittest.mock import patch

import pytest
from app.graph.graph import run_graph, run_graph_full
from app.llm.intent_parser import _heuristic_parse_intent, parse_intent
from app.weather.models import WeatherFacts
from app.weather.open_meteo import WeatherFetchError


def test_heuristic_parser_partial_intents():
    """Verify heuristic parser extracts temporal, location, and activity partials."""
    # Temporal only
    p1 = _heuristic_parse_intent("what about this evening")
    assert p1 is not None
    assert p1.time_context == "this evening"
    assert p1.is_follow_up is True
    assert p1.location is None
    assert p1.activity_categories == []

    # Tomorrow morning
    p2 = _heuristic_parse_intent("what about tomorrow morning?")
    assert p2 is not None
    assert p2.time_context == "tomorrow morning"
    assert p2.is_follow_up is True

    # Location only
    p3 = _heuristic_parse_intent("what about Delhi?")
    assert p3 is not None
    assert p3.location == "Delhi"

    # Activity only
    p4 = _heuristic_parse_intent("what about walking?")
    assert p4 is not None
    assert "outdoor_exercise" in p4.activity_categories
    assert p4.mode == "walking"

    # Truly unrecognized
    p5 = _heuristic_parse_intent("hello there")
    assert p5 is None


def test_1_explicit_location_and_activity():
    """1. Explicit location + activity query."""
    thread_id = str(uuid.uuid4())
    answer, state = run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)
    assert "Bhopal" in (state.get("resolved_location") or state.get("location_text") or "")
    assert state.get("activity") in ["walking", "walk", "outdoor_exercise"]
    assert state.get("requested_time") in ["today", "current"]
    assert state.get("error_type") is None
    assert "Weather Advisory" in answer or "Conditions" in answer or "SOP" in answer or "Safety" in answer


def test_2_time_only_follow_up():
    """2. Time-only follow-up ('What about this evening?')."""
    thread_id = str(uuid.uuid4())
    # Turn 1
    run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)
    # Turn 2: Follow-up
    answer, state = run_graph_full("What about this evening?", thread_id=thread_id)
    assert state.get("error_type") is None
    assert "Service Error" not in answer
    assert "Bhopal" in (state.get("resolved_location") or state.get("location_text") or "")
    assert state.get("activity") in ["walking", "walk", "outdoor_exercise"]
    assert "evening" in (state.get("requested_time") or "").lower()
    assert state.get("weather_facts") is not None


def test_3_location_only_follow_up():
    """3. Location-only follow-up ('What about Delhi?')."""
    thread_id = str(uuid.uuid4())
    # Turn 1
    run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)
    # Turn 2: Follow-up with location change
    answer, state = run_graph_full("What about Delhi?", thread_id=thread_id)
    assert state.get("error_type") is None
    assert "Service Error" not in answer
    assert "Delhi" in (state.get("resolved_location") or state.get("location_text") or "")
    assert state.get("activity") in ["walking", "walk", "outdoor_exercise"]


def test_4_activity_only_follow_up():
    """4. Activity-only follow-up ('What about cycling?')."""
    thread_id = str(uuid.uuid4())
    # Turn 1
    run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)
    # Turn 2: Follow-up with activity change
    answer, state = run_graph_full("What about cycling?", thread_id=thread_id)
    assert state.get("error_type") is None
    assert "Service Error" not in answer
    assert "Bhopal" in (state.get("resolved_location") or state.get("location_text") or "")
    assert state.get("activity") in ["cycling", "outdoor_exercise"]
    assert state.get("intent").mode == "cycling"


def test_5_date_time_follow_up():
    """5. Date + time follow-up ('What about tomorrow morning?')."""
    thread_id = str(uuid.uuid4())
    # Turn 1
    run_graph_full("Is it safe to cycle in Bhopal today?", thread_id=thread_id)
    # Turn 2: Date+time follow-up
    answer, state = run_graph_full("What about tomorrow morning?", thread_id=thread_id)
    assert state.get("error_type") is None
    assert "Service Error" not in answer
    assert "Bhopal" in (state.get("resolved_location") or state.get("location_text") or "")
    assert "tomorrow morning" in (state.get("requested_time") or "").lower()


def test_6_context_replacement():
    """6. Context replacement across multiple turns."""
    thread_id = str(uuid.uuid4())
    # Turn 1: Cycling in Bhopal today
    run_graph_full("Is it safe to cycle in Bhopal today?", thread_id=thread_id)
    # Turn 2: Change activity to walking
    _, state2 = run_graph_full("What about walking?", thread_id=thread_id)
    assert state2.get("activity") in ["walking", "outdoor_exercise"]
    assert state2.get("intent").mode == "walking"
    assert "Bhopal" in (state2.get("resolved_location") or "")

    # Turn 3: Change location and time together
    _, state3 = run_graph_full("What about this evening in Delhi?", thread_id=thread_id)
    assert "Delhi" in (state3.get("resolved_location") or state3.get("location_text") or "")
    assert "evening" in (state3.get("requested_time") or "").lower()
    assert state3.get("activity") in ["walking", "outdoor_exercise"]


def test_7_new_session_incomplete_context():
    """7. New session with incomplete context asking for clarification."""
    thread_id = str(uuid.uuid4())
    answer, state = run_graph_full("What about this evening?", thread_id=thread_id)
    assert state.get("error_type") is None
    assert "Service Error" not in answer
    # Must prompt for clarification
    assert "clarify" in answer.lower() or "where" in answer.lower() or "activity" in answer.lower()


def test_8_gemini_failure_heuristic_fallback():
    """8. Gemini API failure with heuristic fallback continuing successfully."""
    thread_id = str(uuid.uuid4())
    # Turn 1 establishes Bhopal + walking
    run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)

    # Mock parse_intent to raise exception to simulate Gemini API failure/timeout
    with patch("app.graph.nodes.parse_intent.parse_intent", side_effect=Exception("API Timeout")):
        answer, state = run_graph_full("What about this evening?", thread_id=thread_id)
        assert state.get("error_type") is None
        assert "Service Error" not in answer
        assert "Bhopal" in (state.get("resolved_location") or state.get("location_text") or "")
        assert "evening" in (state.get("requested_time") or "").lower()


def test_9_weather_api_failure():
    """9. Weather API failure routes cleanly to weather_failure message."""
    thread_id = str(uuid.uuid4())
    with patch("app.graph.nodes.fetch_weather.fetch_weather", side_effect=WeatherFetchError("Open-Meteo 503")):
        answer, state = run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)
        assert state.get("error_type") == "weather_failure"
        assert "Weather Data Unavailable" in answer


def test_10_no_applicable_sop():
    """10. Match SOPs returning empty routes to no_match_response."""
    thread_id = str(uuid.uuid4())
    with patch("app.graph.nodes.match_sops.match_sops", return_value=[]):
        answer, state = run_graph_full("Is it safe to walk in Bhopal today?", thread_id=thread_id)
        assert state.get("error_type") is None
        assert "No Safety Concerns Identified" in answer or "within normal parameters" in answer
