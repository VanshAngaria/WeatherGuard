"""
Node: compose_answer (kept as utility / fallback)
Renders the standardized response format using template substitution only.
No LLM call. Used as fallback when generate_response LLM call fails,
and directly tested in unit tests.

The full graph uses generate_response_node (LLM narration).
This node produces the same standardized structure deterministically.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)

_SEVERITY_LABEL = {
    "critical": "🚨 CRITICAL",
    "high": "⚠️ HIGH",
    "moderate": "🟡 MODERATE",
    "low": "✅ LOW",
}

_SEVERITY_EMOJI = {
    "critical": "🚨",
    "high": "⚠️",
    "moderate": "⚠️",
    "low": "🌦️",
}


def _format_template(template: str, facts_dict: dict) -> str:
    """
    Fill the advice_template with actual weather values.
    Uses str.format_map with a SafeDict so unknown keys are preserved as-is.
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
    LangGraph node (utility/fallback): compose the final response deterministically.

    Renders the standardized advisory format using template substitution.
    Does NOT call the LLM. Used directly in unit tests and as a fallback
    if generate_response_node fails.
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
    emoji = _SEVERITY_EMOJI.get(primary.severity, "🌦️")
    severity_label = _SEVERITY_LABEL.get(primary.severity, primary.severity.upper())

    # Render advice from template
    recommendation = _format_template(primary.advice_template.strip(), facts_dict)

    # Build conditions block (only non-None facts)
    field_labels = {
        "temperature_2m":            ("🌡️ Temperature",            "°C"),
        "apparent_temperature":      ("🤔 Feels like",              "°C"),
        "relative_humidity_2m":      ("💧 Humidity",                "%"),
        "precipitation_probability": ("🌧️ Rain probability",        "%"),
        "precipitation_sum_today":   ("🌧️ Today's rain total",      " mm"),
        "wind_speed_10m":            ("💨 Wind speed",              " km/h"),
        "wind_gusts_10m":            ("💨 Wind gusts",              " km/h"),
        "uv_index":                  ("☀️ UV index",                ""),
        "visibility":                ("👁️ Visibility",             " m"),
        "weathercode":               ("🌩️ Weather code",            ""),
    }
    condition_lines = []
    for field, (label, unit) in field_labels.items():
        val = facts_dict.get(field)
        if val is not None:
            condition_lines.append(f"  • {label}: **{val}{unit}**")
    conditions_block = "\n".join(condition_lines) if condition_lines else "  • (data not available)"

    # Assemble standardized format
    parts = [
        f"{emoji} **Weather Advisory — {location}**",
        "",
        "**Recommendation**",
        recommendation,
        "",
        "**Current Conditions**",
        conditions_block,
        "",
        "**Severity**",
        severity_label,
        "",
        "**Applicable SOP**",
        f"`{primary.sop_id}` — {primary.sop_title}",
        "",
        "**Why this policy applies**",
        f"Live conditions met the threshold defined in {primary.sop_id}: {', '.join(f'{k}={v}' for k, v in primary.matched_conditions.items())}.",
    ]

    # Secondary matches footnote
    if decision.secondary_matches:
        secondary_ids = ", ".join(m.sop_id for m in decision.secondary_matches)
        parts += [
            "",
            "---",
            f"*Additional policies also triggered: {secondary_ids}. "
            "The primary response above reflects the highest-priority policy.*",
        ]

    final_answer = "\n".join(parts)

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": final_answer}
    ]

    logger.info("compose_answer_node: answer composed, primary=%s", primary.sop_id)

    return {
        "final_answer": final_answer,
        "conversation_history": updated_history,
    }
