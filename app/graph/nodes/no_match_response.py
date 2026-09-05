"""
Node: no_match_response
Produces an honest "no policy found" response when no SOP matches.
Does NOT call the LLM. Does NOT invent safety advice.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)

_NO_MATCH_MSG = (
    "ℹ️ **No Applicable Safety Policy Found**\n\n"
    "I couldn't find an applicable weather-based safety policy for this situation, "
    "so I can't provide a weather-based safety recommendation.\n\n"
    "This may be because:\n"
    "- The current weather conditions in your area are within normal safe ranges for this activity.\n"
    "- This specific activity/scenario is not yet covered by our policy library.\n\n"
    "For general safety guidance, please consult relevant local authorities or activity-specific resources."
)


def no_match_response_node(state: BotState) -> Dict:
    """
    LangGraph node: produce a no-policy-found response.

    - Returns a fixed, honest message.
    - Updates conversation_history.
    - Does NOT call the LLM or invent advice.
    """
    conversation_history = state.get("conversation_history", [])
    location = state.get("resolved_location", "your area")
    facts = state.get("weather_facts")

    logger.info("no_match_response_node: no SOPs matched for location=%s", location)

    answer = _NO_MATCH_MSG

    # Optionally append current weather summary so user knows data was retrieved
    if facts:
        weather_summary_parts = []
        if facts.temperature_2m is not None:
            weather_summary_parts.append(f"🌡️ Temperature: **{facts.temperature_2m}°C**")
        if facts.wind_speed_10m is not None:
            weather_summary_parts.append(f"💨 Wind: **{facts.wind_speed_10m} km/h**")
        if facts.precipitation_probability is not None:
            weather_summary_parts.append(f"🌧️ Rain chance: **{facts.precipitation_probability}%**")

        if weather_summary_parts:
            answer += (
                f"\n\n**Current conditions in {location}:**\n"
                + " | ".join(weather_summary_parts)
            )

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": answer}
    ]

    return {
        "final_answer": answer,
        "conversation_history": updated_history,
    }
