"""
Node: no_match_response
Produces an honest "no safety concern found" response when no SOP matches.
No-match means conditions are within normal safe ranges for known policies,
NOT that the system failed or the activity is unsafe.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)


def no_match_response_node(state: BotState) -> Dict:
    """
    LangGraph node: produce a no-policy-fired response.

    A no-match means: no safety SOP was triggered by current conditions.
    This is generally GOOD news — it means conditions are within normal safe ranges.
    We still show the actual weather numbers so the user can judge for themselves.
    """
    conversation_history = state.get("conversation_history", [])
    location = state.get("resolved_location", "your area")
    facts = state.get("weather_facts")
    intent = state.get("intent")

    logger.info("no_match_response_node: no SOPs matched for location=%s", location)

    # Build weather summary line
    weather_parts = []
    if facts:
        if facts.temperature_2m is not None:
            weather_parts.append(f"🌡️ Temperature: **{facts.temperature_2m}°C**")
        if facts.apparent_temperature is not None:
            weather_parts.append(f"🤔 Feels like: **{facts.apparent_temperature}°C**")
        if facts.relative_humidity_2m is not None:
            weather_parts.append(f"💧 Humidity: **{facts.relative_humidity_2m}%**")
        if facts.wind_speed_10m is not None:
            weather_parts.append(f"💨 Wind: **{facts.wind_speed_10m} km/h**")
        if facts.wind_gusts_10m is not None:
            weather_parts.append(f"💨 Gusts: **{facts.wind_gusts_10m} km/h**")
        if facts.precipitation_probability is not None:
            weather_parts.append(f"🌧️ Rain chance: **{facts.precipitation_probability}%**")
        if facts.uv_index is not None:
            weather_parts.append(f"☀️ UV index: **{facts.uv_index}**")
        if facts.visibility is not None and facts.visibility < 5000:
            weather_parts.append(f"👁️ Visibility: **{facts.visibility} m**")

    # Activity context
    activity_hint = ""
    if intent:
        mode = intent.mode
        cats = intent.activity_categories
        if mode:
            activity_hint = f" for **{mode}**"
        elif cats:
            activity_hint = f" for **{', '.join(cats)}**"

    # Build the answer
    answer_lines = [
        f"✅ **No Weather Safety Concerns Found**\n",
        f"Based on live weather data for **{location}**, no safety thresholds from our policy library were exceeded{activity_hint}.\n",
        f"This means current conditions are within normal safe ranges for all monitored weather hazards (extreme heat, thunderstorm, poor visibility, strong gusts, etc.).\n",
    ]

    # Add weather facts table
    if weather_parts:
        answer_lines.append(f"**Current conditions in {location}:**")
        answer_lines.append(" | ".join(weather_parts))

    # Soft caveats
    answer_lines.append(
        "\n---\n"
        "_This assessment reflects automated safety policies only. "
        "Always use personal judgment and check local conditions before heading out._"
    )

    # Extra note if rain chance is high but no thunderstorm SOP fired
    if facts and facts.precipitation_probability is not None and facts.precipitation_probability >= 60:
        answer_lines.append(
            f"\n⚠️ _Note: Rain probability is **{facts.precipitation_probability}%** — "
            "you may want to carry rain gear even though no formal safety policy was triggered._"
        )

    answer = "\n".join(answer_lines)

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": answer}
    ]

    return {
        "final_answer": answer,
        "conversation_history": updated_history,
    }
