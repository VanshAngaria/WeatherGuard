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
    """Format template string safely preserving unknown keys."""
    class SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}"

    try:
        return template.format_map(SafeDict(facts_dict))
    except Exception as exc:
        logger.warning("Template formatting error: %s", exc)
        return template


def compose_answer_node(state: BotState) -> Dict:
    """Renders the standardized response format deterministically via template substitution."""
    decision = state.get("policy_decision")
    facts = state.get("weather_facts")
    location = state.get("resolved_location", "your location")
    conversation_history = state.get("conversation_history", [])

    if decision is None or facts is None:
        return {
            "final_answer": "I encountered an internal error composing the response. Please try again."
        }

    primary = decision.primary
    facts_dict = facts.to_facts_dict()
    recommendation = _format_template(primary.advice_template, facts_dict)

    # Build conditions block
    cond_lines = []
    field_labels = {
        "temperature_2m": ("🌡️ Temperature", "°C"),
        "apparent_temperature": ("🤔 Feels like", "°C"),
        "relative_humidity_2m": ("💧 Humidity", "%"),
        "precipitation": ("🌧️ Current precipitation", " mm"),
        "precipitation_probability": ("🌧️ Rain probability", "%"),
        "precipitation_sum_today": ("🌧️ Today's rain total", " mm"),
        "wind_speed_10m": ("💨 Wind speed", " km/h"),
        "wind_gusts_10m": ("💨 Wind gusts", " km/h"),
        "uv_index": ("☀️ UV index", ""),
        "visibility": ("👁️ Visibility", " m"),
        "weathercode": ("🌩️ Weather code", ""),
    }
    for field, (label, unit) in field_labels.items():
        val = facts_dict.get(field)
        if val is not None:
            cond_lines.append(f"  • {label}: **{val}{unit}**")

    conditions_block = "\n".join(cond_lines) if cond_lines else "  • (data not available)"
    emoji = _SEVERITY_EMOJI.get(primary.severity, "🌦️")
    severity_label = _SEVERITY_LABEL.get(primary.severity, primary.severity.upper())

    parts = [
        f"{emoji} **Weather Advisory — {location}**",
        "",
        "**Recommendation**",
        recommendation.strip(),
        "",
        "**Current Conditions**",
        conditions_block,
        "",
        f"**Severity**",
        severity_label,
        "",
        "**Applicable SOP**",
        f"`{primary.sop_id}` — {primary.sop_title}",
        "",
        "**Why this policy applies**",
        f"Atmospheric conditions in {location} satisfied the policy criteria for {primary.sop_id}.",
    ]

    if decision.secondary_matches:
        secondary_ids = ", ".join(m.sop_id for m in decision.secondary_matches)
        parts += [
            "",
            "---",
            f"*Additional policies also triggered: {secondary_ids}.*",
        ]

    final_answer = "\n".join(parts)
    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": final_answer}
    ]

    return {
        "final_answer": final_answer,
        "conversation_history": updated_history,
    }
