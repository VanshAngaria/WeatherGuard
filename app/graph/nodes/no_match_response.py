"""
Node: no_match_response
Produces an honest response when no SOP matches.

Two distinct scenarios:
A) ACTIVITY HAS NO SOP COVERAGE
   The user asked about an activity we don't have a policy for.
   → "We don't have a safety SOP for this activity."

B) CONDITIONS ARE WITHIN SAFE RANGES
   The activity is covered but no safety threshold was exceeded.
   → "✅ No safety concerns found — conditions look fine."

Does NOT call the LLM. Does NOT invent safety advice.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.graph.state import BotState
from app.policy.loader import get_sops

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SOP category awareness
# ---------------------------------------------------------------------------

def _any_sop_covers_categories(intent_categories: List[str]) -> bool:
    """
    Return True if at least one loaded SOP applies to any of the intent categories.
    This distinguishes "no SOP for this activity" from "SOP exists but conditions OK".
    """
    sops = get_sops()
    for sop in sops:
        if "*" in sop.applies_to_categories:
            return True
        for cat in intent_categories:
            if cat in sop.applies_to_categories or cat == sop.category:
                return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_weather_inline(facts) -> str:
    """Build a compact weather summary line (pipe-separated)."""
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
    return " | ".join(parts) if parts else "(weather data unavailable)"


def _build_conditions_block(facts) -> str:
    """Build bullet-list conditions block."""
    lines = []
    if not facts:
        return "  • (weather data unavailable)"

    field_labels = {
        "temperature_2m":            ("🌡️ Temperature",           "°C"),
        "apparent_temperature":      ("🤔 Feels like",             "°C"),
        "relative_humidity_2m":      ("💧 Humidity",               "%"),
        "precipitation_probability": ("🌧️ Rain probability",       "%"),
        "precipitation_sum_today":   ("🌧️ Today's rain total",     " mm"),
        "wind_speed_10m":            ("💨 Wind speed",             " km/h"),
        "wind_gusts_10m":            ("💨 Wind gusts",             " km/h"),
        "uv_index":                  ("☀️ UV index",               ""),
        "visibility":                ("👁️ Visibility",            " m"),
    }
    d = facts.to_facts_dict()
    for field, (label, unit) in field_labels.items():
        val = d.get(field)
        if val is not None:
            lines.append(f"  • {label}: **{val}{unit}**")

    return "\n".join(lines) if lines else "  • (weather data unavailable)"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def no_match_response_node(state: BotState) -> Dict:
    """
    LangGraph node: produce a no-policy-fired response.

    Distinguishes between:
    - Activity not covered by any SOP (no guidance available)
    - Activity covered but conditions are fine (positive "all clear")
    """
    conversation_history = state.get("conversation_history", [])
    location = state.get("resolved_location", "your area")
    facts = state.get("weather_facts")
    intent = state.get("intent")

    logger.info("no_match_response_node: no SOPs matched for location=%s", location)

    # --- Activity context ---
    activity_hint = ""
    activity_label = ""
    if intent:
        mode = intent.mode
        cats = intent.activity_categories or []
        if mode:
            activity_hint = f" for **{mode}**"
            activity_label = mode
        elif cats and cats != ["general"]:
            activity_hint = f" for **{', '.join(cats)}**"
            activity_label = cats[0]

    # --- Check if any SOP covers this activity at all ---
    intent_categories = intent.activity_categories if intent else []
    # Filter out 'general' — it's a fallback category, not a real activity signal
    real_cats = [c for c in intent_categories if c != "general"]
    has_coverage = _any_sop_covers_categories(real_cats) if real_cats else True

    conditions_block = _build_conditions_block(facts)

    if not has_coverage and real_cats:
        # --- SCENARIO A: No SOP coverage for this activity ---
        answer_lines = [
            f"🌤️ **Weather Advisory — {location}**",
            "",
            "**No Safety Policy Available**",
            "",
            f"I checked the live weather data{activity_hint}, but our policy library "
            f"doesn't have a safety SOP for this specific activity. "
            f"I can't provide a policy-based recommendation without an applicable SOP.",
            "",
            "**Current Conditions**",
            conditions_block,
            "",
            "---",
            "_To get a policy-based advisory, ask about: cycling, walking, running, "
            "outdoor recreation, travel, or activities for children or elderly._",
        ]
    else:
        # --- SCENARIO B: Conditions are within safe ranges ---
        answer_lines = [
            f"✅ **Weather Advisory — {location}**",
            "",
            "**Recommendation**",
            f"No safety concerns found{activity_hint}. "
            f"Current conditions in **{location}** are within normal safe ranges — "
            f"none of our weather safety thresholds were exceeded.",
            "",
            "**Current Conditions**",
            conditions_block,
            "",
            "**Severity**",
            "✅ NO SAFETY CONCERNS",
            "",
            "---",
            "_This assessment reflects automated safety policies only. "
            "Always use personal judgment and check local conditions before heading out._",
        ]

        # Extra advisory if rain chance is high but no SOP fired
        if facts and facts.precipitation_probability is not None and facts.precipitation_probability >= 60:
            answer_lines.append(
                f"\n⚠️ _Note: Rain probability is **{facts.precipitation_probability}%** — "
                "consider carrying rain gear even though no formal safety policy was triggered._"
            )

    answer = "\n".join(answer_lines)

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": answer}
    ]

    return {
        "final_answer": answer,
        "conversation_history": updated_history,
    }
