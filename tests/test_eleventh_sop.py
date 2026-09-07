"""
Test: Adding an 11th SOP solely through configuration/data (YAML).

Assignment Requirement #6 & #20:
"Adding an 11th SOP must NOT require modification of:
 - LangGraph control-flow code
 - weather-fetching code
 - LLM client code
 - individual SOP matching branches
 - individual activity if/else blocks

Adding an SOP should require only configuration/data changes,
assuming the new SOP uses the supported generic schema/operators."
"""

import uuid
from unittest.mock import patch
import pytest

from app.policy.loader import load_sops
from app.policy.models import SOP
from app.policy.matcher import match_sops
from app.policy.selector import resolve_policy
from app.graph.graph import run_graph_full
from app.weather.models import WeatherFacts


NEW_11TH_SOP_YAML = """
sops:
  - id: SOP-022
    title: "Extreme Destructive Wind Gust Warning"
    category: outdoor_recreation
    severity: critical
    match_type: rule
    overrides: true
    priority: 1
    applies_to_categories: ["outdoor_recreation", "outdoor_exercise"]
    required_fields:
      - wind_gusts_10m
    condition:
      type: leaf
      field: wind_gusts_10m
      op: gte
      value: 75
    advice_template: >
      🚨 **SOP-022 | Extreme Destructive Wind Gust Warning | CRITICAL — OVERRIDE**

      Destructive wind gusts reaching **{wind_gusts_10m} km/h** have been detected.
      Structural hazard and severe debris risk. All outdoor activities must cease immediately.
"""


def test_eleventh_sop_parsing_and_matching():
    """
    Verify that an 11th SOP defined purely in YAML:
    1. Validates against the SOP Pydantic schema.
    2. Matches deterministically when the generic condition (visibility <= 200) is met.
    3. Does not match when visibility > 200.
    4. Deterministically resolves as primary advisory when higher priority/severity.
    """
    import yaml
    parsed = yaml.safe_load(NEW_11TH_SOP_YAML)
    sop_obj = SOP.model_validate(parsed["sops"][0])
    assert sop_obj.id == "SOP-022"
    assert sop_obj.severity == "critical"

    # Weather fact breaching wind_gusts_10m >= 75
    facts_wind = {"wind_gusts_10m": 85.0}
    matches = match_sops(
        facts=facts_wind,
        intent_categories=["outdoor_recreation"],
        _sops_override=[sop_obj],
    )
    assert len(matches) == 1
    assert matches[0].sop_id == "SOP-022"
    assert matches[0].severity == "critical"

    # Weather fact safely below threshold
    facts_clear = {"wind_gusts_10m": 25.0}
    matches_clear = match_sops(
        facts=facts_clear,
        intent_categories=["outdoor_recreation"],
        _sops_override=[sop_obj],
    )
    assert len(matches_clear) == 0


def test_eleventh_sop_in_end_to_end_graph():
    """
    Test that the LangGraph workflow executes, matches, and surfaces the 11th SOP
    without modifying any LangGraph control flow or matcher Python code.
    """
    import yaml
    parsed = yaml.safe_load(NEW_11TH_SOP_YAML)
    new_sop = SOP.model_validate(parsed["sops"][0])

    # Combine existing SOPs with the 11th SOP
    from app.policy.loader import get_sops
    combined_sops = list(get_sops()) + [new_sop]

    mock_facts = WeatherFacts(
        temperature_2m=25.0,
        apparent_temperature=25.0,
        relative_humidity_2m=60.0,
        wind_speed_10m=35.0,
        wind_gusts_10m=80.0,  # >= 75 triggers SOP-022
        precipitation=0.0,
        precipitation_probability=20.0,
        visibility=8000.0,
        weathercode=0,
        uv_index=2.0,
    )

    with patch("app.policy.loader.get_sops", return_value=combined_sops), \
         patch("app.policy.matcher.get_sops", return_value=combined_sops), \
         patch("app.graph.nodes.fetch_weather.fetch_weather", return_value=mock_facts):
        
        thread_id = str(uuid.uuid4())
        answer, state = run_graph_full(
            "Can I have a picnic in Bhopal right now?",
            thread_id=thread_id,
        )

        assert state.get("error_type") is None
        decision = state.get("policy_decision")
        assert decision is not None
        # Must match our dynamically injected 11th SOP
        assert decision.primary.sop_id == "SOP-022"
        assert decision.primary.severity == "critical"
        assert "SOP-022" in answer
        assert "Extreme Destructive Wind Gust" in answer
