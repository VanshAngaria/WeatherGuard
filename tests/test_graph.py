"""
Graph integration tests.
Tests routing logic with mocked weather and geocoding.
Does NOT require OPENAI_API_KEY for routing tests (intent is also mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.compose_answer import compose_answer_node
from app.graph.nodes.error_response import error_response_node
from app.graph.nodes.match_sops import match_sops_node
from app.graph.nodes.no_match_response import no_match_response_node
from app.graph.nodes.resolve_policy import resolve_policy_node
from app.graph.state import BotState
from app.llm.intent_parser import ParsedIntent
from app.policy.matcher import match_sops
from app.weather.models import WeatherFacts


def _make_facts(**kwargs) -> WeatherFacts:
    return WeatherFacts(**kwargs)


def _make_intent(categories=None, mode=None, group=None, location="London") -> ParsedIntent:
    return ParsedIntent(
        activity_categories=categories or ["outdoor_exercise"],
        mode=mode,
        group=group,
        location=location,
        time_context="current",
        is_follow_up=False,
    )


# ---------------------------------------------------------------------------
# match_sops_node
# ---------------------------------------------------------------------------

def test_match_sops_thunderstorm():
    """Thunderstorm weathercode must match SOP-005."""
    facts = _make_facts(weathercode=95, apparent_temperature=25.0)
    intent = _make_intent(categories=["outdoor_exercise"], mode="running")
    state: BotState = {"weather_facts": facts, "intent": intent}

    result = match_sops_node(state)
    matches = result["sop_matches"]
    ids = [m.sop_id for m in matches]
    assert "SOP-005" in ids


def test_match_sops_no_match():
    """Normal benign weather produces no matches for water_activities."""
    facts = _make_facts(
        temperature_2m=22.0,
        apparent_temperature=23.0,
        weathercode=1,
        visibility=10000.0,
        wind_gusts_10m=10.0,
        precipitation_probability=5.0,
        wind_speed_10m=8.0,
        uv_index=4.0,
        precipitation_sum_today=0.0,
        precipitation_sum_next_2d=0.0,
    )
    intent = _make_intent(categories=["water_activities"], mode="swimming")
    state: BotState = {"weather_facts": facts, "intent": intent}

    result = match_sops_node(state)
    assert result["sop_matches"] == []


# ---------------------------------------------------------------------------
# resolve_policy_node
# ---------------------------------------------------------------------------

def test_resolve_policy_override_wins():
    """When SOP-005 (override) and SOP-001 both match, SOP-005 is primary."""
    from app.policy.models import MatchResult

    m1 = MatchResult(
        sop_id="SOP-001", sop_title="Extreme Heat", category="outdoor_exercise",
        severity="critical", overrides=False, priority=10,
        matched_conditions={"apparent_temperature": 44},
        advice_template="Heat advice.",
    )
    m2 = MatchResult(
        sop_id="SOP-005", sop_title="Thunderstorm", category="outdoor_recreation",
        severity="critical", overrides=True, priority=1,
        matched_conditions={"weathercode": 95},
        advice_template="Thunder advice.",
    )
    state: BotState = {"sop_matches": [m1, m2]}
    result = resolve_policy_node(state)
    decision = result["policy_decision"]
    assert decision.primary.sop_id == "SOP-005"
    assert any(m.sop_id == "SOP-001" for m in decision.secondary_matches)


# ---------------------------------------------------------------------------
# error_response_node
# ---------------------------------------------------------------------------

def test_error_response_location_failure():
    state: BotState = {
        "error_type": "location_failure",
        "error": "No results found",
        "conversation_history": [],
    }
    result = error_response_node(state)
    assert "Location Not Found" in result["final_answer"]


def test_error_response_weather_failure():
    state: BotState = {
        "error_type": "weather_failure",
        "error": "Timeout",
        "conversation_history": [],
    }
    result = error_response_node(state)
    assert "Weather Data Unavailable" in result["final_answer"]


# ---------------------------------------------------------------------------
# no_match_response_node
# ---------------------------------------------------------------------------

def test_no_match_response():
    facts = _make_facts(temperature_2m=22.0, wind_speed_10m=10.0)
    state: BotState = {
        "weather_facts": facts,
        "resolved_location": "London",
        "conversation_history": [],
    }
    result = no_match_response_node(state)
    answer = result["final_answer"]
    # New format: standardized header + no-concerns message
    assert "Weather Advisory" in answer
    assert "No safety concerns" in answer or "No Safety Policy" in answer


# ---------------------------------------------------------------------------
# compose_answer_node — template rendering
# ---------------------------------------------------------------------------

def test_compose_answer_template_fills_values():
    """Standardized format contains SOP ID, severity, location, and actual weather value."""
    from app.policy.models import MatchResult, PolicyDecision

    primary = MatchResult(
        sop_id="SOP-001",
        sop_title="Extreme Heat",
        category="outdoor_exercise",
        severity="critical",
        overrides=False,
        priority=10,
        matched_conditions={"apparent_temperature": 44.0},
        advice_template="**SOP-001 | Extreme Heat | CRITICAL** Apparent temperature is {apparent_temperature}°C.",
    )
    decision = PolicyDecision(primary=primary, secondary_matches=[], resolution_reason="test")
    facts = _make_facts(apparent_temperature=44.0)

    state: BotState = {
        "policy_decision": decision,
        "weather_facts": facts,
        "resolved_location": "Test City",
        "conversation_history": [],
    }
    result = compose_answer_node(state)
    answer = result["final_answer"]
    # Standardized format checks
    assert "SOP-001" in answer
    assert "Test City" in answer
    assert "CRITICAL" in answer
    assert "44.0" in answer          # actual weather value present
    assert "Weather Advisory" in answer  # standardized header


# ---------------------------------------------------------------------------
# Full graph routing tests (with mocked LLM + weather)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_graph_routes_to_no_match_on_benign_weather():
    """
    Full graph: benign weather + water_activities → no_match_response.
    Mocks LLM intent and weather API.
    """
    from app.graph.graph import build_graph

    intent = _make_intent(categories=["water_activities"], mode="swimming", location="London")

    mock_facts = _make_facts(
        temperature_2m=20.0,
        apparent_temperature=21.0,
        weathercode=1,
        visibility=10000.0,
        wind_gusts_10m=5.0,
        precipitation_probability=5.0,
        wind_speed_10m=5.0,
        uv_index=3.0,
        precipitation_sum_today=0.0,
        precipitation_sum_next_2d=0.0,
    )

    with (
        patch("app.graph.nodes.parse_intent.parse_intent", return_value=intent),
        patch("app.graph.nodes.resolve_location.resolve_location", return_value=(51.5, -0.1, "London, England, UK")),
        patch("app.graph.nodes.fetch_weather.fetch_weather", return_value=mock_facts),
    ):
        graph = build_graph()
        result = graph.invoke(
            {"user_message": "Is it safe to swim today?", "thread_id": "test-001"},
            config={"configurable": {"thread_id": "test-001"}},
        )

    assert "Weather Advisory" in result["final_answer"]
    assert "No safety concerns" in result["final_answer"] or "No Safety Policy" in result["final_answer"]


@pytest.mark.integration
def test_graph_routes_to_error_on_weather_failure():
    """Full graph: weather API failure → error_response."""
    from app.graph.graph import build_graph
    from app.weather.open_meteo import WeatherFetchError

    intent = _make_intent(categories=["outdoor_exercise"])

    with (
        patch("app.graph.nodes.parse_intent.parse_intent", return_value=intent),
        patch("app.graph.nodes.resolve_location.resolve_location", return_value=(51.5, -0.1, "London")),
        patch("app.graph.nodes.fetch_weather.fetch_weather", side_effect=WeatherFetchError("timeout")),
    ):
        graph = build_graph()
        result = graph.invoke(
            {"user_message": "Is it safe to run?", "thread_id": "test-002"},
            config={"configurable": {"thread_id": "test-002"}},
        )

    assert "Weather Data Unavailable" in result["final_answer"]


@pytest.mark.integration
def test_graph_routes_to_generate_response_on_sop_match():
    """Full graph: thunderstorm weather + outdoor_exercise → SOP-005 → generate_response."""
    from app.graph.graph import build_graph

    intent = _make_intent(categories=["outdoor_exercise"], mode="cycling", location="London")

    mock_facts = _make_facts(
        weathercode=95,
        temperature_2m=28.0,
        apparent_temperature=30.0,
        wind_gusts_10m=40.0,
        precipitation_probability=80.0,
        wind_speed_10m=25.0,
        visibility=800.0,
        uv_index=3.0,
        precipitation_sum_today=20.0,
        precipitation_sum_next_2d=30.0,
    )

    # Mock generate_response_node's LLM call so we don't need a real API key in CI
    mock_llm_response = MagicMock()
    mock_llm_response.text = '{"recommendation": "Do not cycle outdoors.", "why": "Thunderstorm is active."}'

    with (
        patch("app.graph.nodes.parse_intent.parse_intent", return_value=intent),
        patch("app.graph.nodes.resolve_location.resolve_location", return_value=(51.5, -0.1, "London, UK")),
        patch("app.graph.nodes.fetch_weather.fetch_weather", return_value=mock_facts),
        patch("app.graph.nodes.generate_response.get_llm_client") as mock_client,
    ):
        mock_client.return_value.models.generate_content.return_value = mock_llm_response
        graph = build_graph()
        result = graph.invoke(
            {"user_message": "Is it safe to cycle in London?", "thread_id": "test-003"},
            config={"configurable": {"thread_id": "test-003"}},
        )

    answer = result["final_answer"]
    assert "SOP-005" in answer           # correct SOP identified
    assert "CRITICAL" in answer          # severity preserved
    assert "Weather Advisory" in answer  # standardized format header
    assert "London" in answer            # location present


@pytest.mark.integration
def test_sop_015_matches_high_rain_probability():
    """SOP-015 fires when precipitation_probability >= 70 for outdoor_exercise."""
    facts = _make_facts(precipitation_probability=75.0)
    intent = _make_intent(categories=["outdoor_exercise"], mode="cycling")
    state: BotState = {"weather_facts": facts, "intent": intent}

    result = match_sops_node(state)
    matches = result["sop_matches"]
    ids = [m.sop_id for m in matches]
    assert "SOP-015" in ids, f"Expected SOP-015 in {ids}"
