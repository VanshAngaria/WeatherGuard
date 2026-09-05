"""
Node: error_response
Produces a fixed, honest failure response when weather or location fetch fails.
Does NOT call the LLM. Does NOT invent data.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)

_LOCATION_FAILURE_MSG = (
    "❌ **Location Not Found**\n\n"
    "I couldn't resolve that location, so I can't retrieve the live weather data "
    "needed for a safety recommendation.\n\n"
    "Please try providing a more specific place name — for example, "
    "\"Mumbai, India\" or \"London, UK\"."
)

_WEATHER_FAILURE_MSG = (
    "❌ **Weather Data Unavailable**\n\n"
    "I couldn't retrieve live weather data for this location, so I can't provide "
    "a weather-based safety recommendation right now.\n\n"
    "This may be a temporary issue with the weather service. Please try again in a few moments."
)

_LLM_FAILURE_MSG = (
    "❌ **Service Error**\n\n"
    "I encountered an error understanding your request. Please try rephrasing it, "
    "or try again in a moment."
)

_GENERIC_FAILURE_MSG = (
    "❌ **Unexpected Error**\n\n"
    "Something went wrong. Please try again."
)


def error_response_node(state: BotState) -> Dict:
    """
    LangGraph node: produce a failure response.

    - Reads error_type to select the appropriate canned message.
    - Updates conversation_history.
    - Returns final_answer.
    """
    error_type = state.get("error_type", "")
    error_detail = state.get("error", "")
    conversation_history = state.get("conversation_history", [])

    logger.warning("error_response_node: type=%s detail=%s", error_type, error_detail)

    if error_type == "location_failure":
        answer = _LOCATION_FAILURE_MSG
    elif error_type == "weather_failure":
        answer = _WEATHER_FAILURE_MSG
    elif error_type == "llm_failure":
        answer = _LLM_FAILURE_MSG
    else:
        answer = _GENERIC_FAILURE_MSG

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": answer}
    ]

    return {
        "final_answer": answer,
        "conversation_history": updated_history,
    }
