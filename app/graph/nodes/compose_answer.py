"""
Node: compose_answer
Renders the final user-facing response from the SOP advice template.
Template substitution uses only WeatherFacts values — no LLM generation.

The LLM does NOT decide the advice content.
The primary SOP's advice_template drives the response.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)


def _format_template(template: str, facts_dict: dict) -> str:
    """
    Fill the advice_template with actual weather values.
    Uses Python str.format_map with a defaultdict-like fallback so
    unknown keys are preserved as-is rather than raising KeyError.
    """

    class SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}"

    try:
        return template.format_map(SafeDict(facts_dict))
    except Exception as exc:
        logger.warning("Template formatting error: %s", exc)
        return template


def compose_answer_node(state: BotState) -> Dict:
    """
    LangGraph node: compose the final response.

    - Reads policy_decision and weather_facts from state.
    - Renders the primary SOP's advice_template with actual weather values.
    - Appends secondary match information.
    - Updates conversation_history with the assistant response.
    - Returns final_answer.
    """
    decision = state.get("policy_decision")
    facts = state.get("weather_facts")
    location = state.get("resolved_location", "your location")
    conversation_history = state.get("conversation_history", [])

    if decision is None or facts is None:
        logger.error("compose_answer_node: policy_decision or weather_facts missing.")
        return {
            "final_answer": (
                "I encountered an internal error composing the response. "
                "Please try again."
            )
        }

    primary = decision.primary
    facts_dict = facts.to_facts_dict()

    # Render primary advice
    answer_parts = [
        f"**📍 Location:** {location}\n",
        _format_template(primary.advice_template.strip(), facts_dict),
    ]

    # Append secondary matches as informational footnotes
    if decision.secondary_matches:
        secondary_ids = ", ".join(m.sop_id for m in decision.secondary_matches)
        answer_parts.append(
            f"\n\n---\n*Additional policies also triggered: {secondary_ids}. "
            f"The primary response above reflects the highest-priority policy.*"
        )

    final_answer = "\n".join(answer_parts)

    # Update conversation history with assistant response
    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": final_answer}
    ]

    logger.info("compose_answer_node: answer composed, primary=%s", primary.sop_id)

    return {
        "final_answer": final_answer,
        "conversation_history": updated_history,
    }
