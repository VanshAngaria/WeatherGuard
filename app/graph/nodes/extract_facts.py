"""
Node: extract_facts
WeatherFacts is already produced by fetch_weather_node.
This node is a lightweight pass-through that validates and logs the facts,
maintaining clean node separation in the graph.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)


def extract_facts_node(state: BotState) -> Dict:
    """
    LangGraph node: validate and log extracted WeatherFacts.

    The actual extraction happens in fetch_weather_node.
    This node validates that WeatherFacts is present and logs a summary.
    """
    facts = state.get("weather_facts")

    if facts is None:
        logger.error("extract_facts_node: WeatherFacts is None — unexpected state.")
        return {
            "error": "Weather facts could not be extracted.",
            "error_type": "weather_failure",
        }

    non_none = {k: v for k, v in facts.model_dump().items() if v is not None}
    logger.info("WeatherFacts available: %s", list(non_none.keys()))

    # No state changes — facts already in state
    return {}
