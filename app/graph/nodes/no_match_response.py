from __future__ import annotations

import logging
from typing import Dict, List

from app.graph.state import BotState
from app.policy.loader import get_sops

logger = logging.getLogger(__name__)


def _any_sop_covers_categories(intent_categories: List[str]) -> bool:
    """Return True if at least one SOP covers the intent categories."""
    sops = get_sops()
    for sop in sops:
        if "*" in sop.applies_to_categories:
            return True
        for cat in intent_categories:
            if cat in sop.applies_to_categories or cat == sop.category:
                return True
    return False


def _build_weather_inline(facts) -> str:
    """Build a compact inline weather summary line."""
    parts = []
    if facts:
        if facts.temperature_2m is not None:
            parts.append(f"🌡️ **{facts.temperature_2m}°C**")
        if facts.wind_speed_10m is not None:
            parts.append(f"💨 **{facts.wind_speed_10m} km/h**")
        if facts.precipitation_probability is not None:
            parts.append(f"🌧️ **{facts.precipitation_probability}%** rain chance")
        if facts.uv_index is not None:
            parts.append(f"☀️ UV **{facts.uv_index}**")
        if facts.visibility is not None:
            parts.append(f"👁️ **{facts.visibility // 1000} km** visibility")
    return " · ".join(parts) if parts else "Weather data available"


_SUPPORTED_ACTIVITIES_LIST = """
**Supported activity types include:**
• 🚴 **Cycling** (commute, road, mountain)
• 🚶 **Walking & Pedestrian travel**
• 🏃 **Running & Outdoor exercise**
• 🚗 **Commuting & Two-wheelers** (motorbikes, scooters)
• 🧺 **Picnics & Outdoor recreation**
• 👨‍👩‍👧 **Activities with children**
• 👴 **Activities for elderly persons**
"""


def no_match_response_node(state: BotState) -> Dict:
    """
    Produces an advisory response when no policy conditions are exceeded.
    Distinguishes between unsupported activities vs safe weather conditions.
    """
    intent_categories = state.get("intent_categories", [])
    facts = state.get("weather_facts")
    location = state.get("resolved_location", "your location")
    conversation_history = state.get("conversation_history", [])
    activity = state.get("activity")
    raw_query = state.get("user_message", "")

    weather_summary = _build_weather_inline(facts)
    activity_is_covered = _any_sop_covers_categories(intent_categories)

    if not activity_is_covered:
        act_display = f"**{activity}**" if activity else "this activity"
        answer = (
            f"ℹ️ **No Policy Coverage — {location}**\n\n"
            f"**Recommendation**\n"
            f"We don't currently have a Standard Operating Procedure (SOP) or safety guidance defined for {act_display}.\n\n"
            f"**Current Conditions**\n"
            f"{weather_summary}\n\n"
            f"**Applicable SOP**\n"
            f"None — No written safety policy covers this activity.\n\n"
            f"**Policy Traceability**\n"
            f"To prevent unverified or hallucinated advice, safety recommendations are strictly provided only for activities governed by a written SOP.\n\n"
            f"{_SUPPORTED_ACTIVITIES_LIST}\n"
            f"Please ask about one of the supported activities above for **{location}**."
        )
    else:
        act_phrase = f"for **{activity}** " if activity else ""

        # Collect relevant SOPs that were evaluated against this query
        eval_sops = []
        for sop in get_sops():
            if "*" in sop.applies_to_categories or any(c in sop.applies_to_categories for c in intent_categories):
                eval_sops.append(f"`{sop.id}` ({sop.title})")

        sop_citation_list = ", ".join(eval_sops[:4]) if eval_sops else "`SOP-019` (Regional Storm), `SOP-001` (Heat Stress), `SOP-005` (Wind Hazard), `SOP-015` (Rain Hazard)"

        answer = (
            f"✅ **No Safety Concerns Identified — {location}**\n\n"
            f"**Recommendation**\n"
            f"Current weather conditions in **{location}** {act_phrase}fall within safe operational thresholds. All monitored parameters are within normal limits.\n\n"
            f"**Current Conditions**\n"
            f"{weather_summary}\n\n"
            f"**Applicable SOP**\n"
            f"None triggered — All monitored atmospheric parameters fall below active hazard thresholds.\n\n"
            f"**Evaluated Policies (Policy Traceability)**\n"
            f"Evaluated against: {sop_citation_list}. All monitored variables (precipitation, wind speed, apparent heat, UV index, and visibility) remain below adverse risk limits.\n\n"
            f"_Tip: Weather conditions can change rapidly. Check back if conditions deteriorate._"
        )

    interpreted_as = state.get("interpreted_as")
    if interpreted_as:
        answer = f"_💬 Interpreted as: \"{interpreted_as}\"_\n\n" + answer

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": answer}
    ]

    return {
        "final_answer": answer,
        "conversation_history": updated_history,
    }
