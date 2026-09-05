"""
LangGraph state definition.
Shared state dict that flows between all graph nodes.
All fields are Optional to allow partial population at each stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from app.llm.intent_parser import ParsedIntent
from app.policy.models import MatchResult, PolicyDecision
from app.weather.models import WeatherFacts


class BotState(TypedDict, total=False):
    """
    Full shared state for one conversation turn through the LangGraph graph.

    Lifecycle:
      parse_intent        → populates: intent, activity, requested_time,
                            needs_clarification, conversation_history, error
      resolve_location    → populates: lat, lon, resolved_location, error
      fetch_weather       → populates: raw_weather_response, error
      extract_facts       → populates: weather_facts
      match_sops          → populates: sop_matches
      resolve_policy      → populates: policy_decision, selected_sop
      generate_response   → populates: final_answer
      error_response      → populates: final_answer
      no_match_response   → populates: final_answer
      clarification_response → populates: final_answer
    """

    # --- Input ---
    user_message: str                       # current user turn
    thread_id: str                          # session identifier

    # --- Conversation memory ---
    conversation_history: List[Dict]        # list of {role, content}

    # --- Intent ---
    intent: Optional[ParsedIntent]

    # --- Resolved structured context (explicit, not buried in intent blob) ---
    activity: Optional[str]                 # e.g. "cycling", "park visit with child"
    requested_time: Optional[str]           # e.g. "evening", "tomorrow morning"
    needs_clarification: Optional[bool]     # True → route to clarification_response

    # --- Input normalization ---
    interpreted_as: Optional[str]           # normalized version of user_message if changed
    scope_type: Optional[str]               # "relevant" | "irrelevant" | "ambiguous" | "no_sop_activity"

    # --- Location ---
    location_text: Optional[str]            # raw location string from intent / memory
    lat: Optional[float]
    lon: Optional[float]
    resolved_location: Optional[str]        # canonical name from geocoding

    # --- Weather ---
    raw_weather_response: Optional[Dict[str, Any]]  # raw API JSON (for debugging)
    weather_facts: Optional[WeatherFacts]

    # --- Policy ---
    sop_matches: Optional[List[MatchResult]]
    policy_decision: Optional[PolicyDecision]
    selected_sop: Optional[MatchResult]     # the winning SOP (primary from decision)

    # --- Output ---
    final_answer: Optional[str]

    # --- Error ---
    error: Optional[str]                    # human-readable error reason
    error_type: Optional[str]              # "location_failure" | "weather_failure" | "llm_failure"
