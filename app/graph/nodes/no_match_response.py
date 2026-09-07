from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.graph.state import BotState
from app.policy.loader import get_sops

logger = logging.getLogger(__name__)


def _build_conditions_block(facts) -> str:
    """Build formatted bullet points from non-empty weather facts."""
    lines = []
    if not facts:
        return "  • (weather data not available)"

    d = facts.to_facts_dict()
    field_labels = {
        "temperature_2m":           ("🌡️ Temperature",           "°C"),
        "apparent_temperature":     ("🤔 Feels like",             "°C"),
        "relative_humidity_2m":     ("💧 Humidity",               "%"),
        "precipitation":            ("🌧️ Current precipitation",  " mm"),
        "precipitation_probability":("🌧️ Rain probability",       "%"),
        "precipitation_sum_today":  ("🌧️ Today's rain total",     " mm"),
        "wind_speed_10m":           ("💨 Wind speed",             " km/h"),
        "wind_gusts_10m":           ("💨 Wind gusts",             " km/h"),
        "uv_index":                 ("☀️ UV index",               ""),
        "visibility":               ("👁️ Visibility",            " m"),
        "weathercode":              ("🌩️ Weather code",           ""),
    }

    for field, (label, unit) in field_labels.items():
        val = d.get(field)
        if val is not None:
            lines.append(f"  • {label}: **{val}{unit}**")

    return "\n".join(lines) if lines else "  • (weather data not available)"


def _get_applicable_sops_for_categories(intent_categories: List[str]):
    """Return all SOPs from the policy library applicable to the specified intent categories."""
    sops = get_sops()
    applicable = []
    for sop in sops:
        if intent_categories == ["general"]:
            if "*" in sop.applies_to_categories or "general" in sop.applies_to_categories or sop.category == "general":
                applicable.append(sop)
        else:
            if any(c in sop.applies_to_categories or c == sop.category for c in intent_categories):
                applicable.append(sop)
    return applicable


def _get_all_supported_categories_summary() -> str:
    """Dynamically summarize all supported activity domains from loaded SOP definitions."""
    sops = get_sops()
    unique_cats = sorted({sop.category for sop in sops if sop.category != "general"})
    cat_bullets = [f"• **{c.replace('_', ' ').title()}**" for c in unique_cats]
    return "**Supported activity categories defined in policy catalog:**\n" + "\n".join(cat_bullets)


def no_match_response_node(state: BotState) -> Dict:
    """
    Produces a policy-governed advisory response when no safety conditions are exceeded
    or when an unsupported activity is requested.

    Deterministic Guarantee:
    - Zero LLM calls to invent safety advice or generic recommendations.
    - Case A: Monitored safety policies apply to the activity/category, and all atmospheric
      parameters remain safely below hazard limits.
    - Case B: No applicable SOP exists in the policy library for this activity. Explicitly
      refuses to claim 'safe' or invent generic advice.
    """
    intent = state.get("intent")
    intent_categories = state.get("intent_categories") or (intent.activity_categories if intent else [])
    facts = state.get("weather_facts")
    location = state.get("resolved_location", "your location")
    conversation_history = state.get("conversation_history", [])
    activity = state.get("activity")
    user_query = state.get("user_message", "")
    time_label = getattr(facts, "time_label", None) or "Current Conditions"

    # Filter out 'general' to verify if a specific unsupported activity was asked
    real_cats = [c for c in intent_categories if c != "general"]
    applicable_sops = _get_applicable_sops_for_categories(real_cats) if real_cats else _get_applicable_sops_for_categories(["general"])

    # Indoor / Unsupported activity check
    # Check if the categories specifically lack coverage
    is_indoor_activity = "indoor_activity" in (real_cats or [])
    has_coverage = len(applicable_sops) > 0 and not is_indoor_activity

    conditions_block = _build_conditions_block(facts)
    subtitle = f"_{activity.title()} · {time_label}_" if activity and activity != "general" else f"_{time_label}_"

    # =========================================================================
    # CASE B: NO APPLICABLE POLICY COVERAGE
    # Requirement: Explicitly state that no applicable policy exists. DO NOT say "safe".
    # =========================================================================
    if not has_coverage:
        act_display = f"**{activity}**" if activity else "this activity"
        supported_summary = _get_all_supported_categories_summary()

        if is_indoor_activity:
            answer = (
                f"ℹ️ **No Applicable Weather Policy — {location}**\n"
                f"{subtitle}\n\n"
                f"**Recommendation**\n"
                f"No weather-safety policy covers indoor activities such as {act_display}. "
                f"Since this is an indoor environment, outdoor weather conditions generally do not directly impact safety.\n\n"
                f"If you were asking about traveling to the venue (e.g., walking or cycling there) "
                f"or outdoor workouts, please specify the outdoor transit mode.\n\n"
                f"**Severity**\n"
                f"ℹ️ NO APPLICABLE POLICY\n\n"
                f"**Applicable SOP**\n"
                f"None — No standard operating procedure covers indoor activities.\n\n"
                f"**Policy Traceability**\n"
                f"Evaluated category `indoor_activity` against loaded SOP catalog. Zero matching procedures exist.\n\n"
                f"{supported_summary}\n\n"
                f"Please ask about one of the supported outdoor activities above for **{location}**."
            )
        else:
            answer = (
                f"ℹ️ **No Applicable Policy — {location}**\n"
                f"{subtitle}\n\n"
                f"**Recommendation**\n"
                f"I don't have an applicable weather-safety policy for {act_display}. "
                f"To prevent unverified or hallucinated advice, safety recommendations are strictly provided only for activities governed by verified Standard Operating Procedures (SOPs).\n\n"
                f"**{time_label}**\n"
                f"{conditions_block}\n\n"
                f"**Severity**\n"
                f"ℹ️ NO APPLICABLE POLICY\n\n"
                f"**Applicable SOP**\n"
                f"None — No standard operating procedure covers {act_display}.\n\n"
                f"**Policy Traceability**\n"
                f"Evaluated categories `{real_cats or ['general']}` against loaded SOP catalog. Zero matching procedures exist for this activity.\n\n"
                f"{supported_summary}\n\n"
                f"Please ask about one of the supported activities above for **{location}**."
            )

    # =========================================================================
    # CASE A: POLICY COVERAGE EXISTS, ALL THRESHOLDS SAFE
    # Deterministic composition: All monitored SOP thresholds verified clear.
    # =========================================================================
    else:
        eval_bullets = [
            f"• `{sop.id}` ({sop.title}) — Within safe limits ✅"
            for sop in applicable_sops
        ]
        eval_names = [f"{sop.id} ({sop.title})" for sop in applicable_sops]
        evaluated_block = "\n".join(eval_bullets[:8])

        act_phrase = f"for **{activity}** " if activity and activity != "general" else ""
        recommendation = (
            f"Current weather conditions in **{location}** {act_phrase}fall within safe operational thresholds. "
            f"None of the adverse safety thresholds monitored in our policy library were triggered."
        )
        why = (
            f"Live atmospheric observations were verified against applicable safety policies: "
            f"{', '.join(eval_names[:4])}. Monitored parameters remain within safe limits."
        )

        answer = (
            f"✅ **No Safety Concerns Identified — {location}**\n"
            f"{subtitle}\n\n"
            f"**Recommendation**\n"
            f"{recommendation}\n\n"
            f"**{time_label}**\n"
            f"{conditions_block}\n\n"
            f"**Severity**\n"
            f"✅ LOW (SAFE)\n\n"
            f"**Applicable SOP**\n"
            f"All Monitored Policies Clear — Safe Operational Thresholds Verified\n\n"
            f"**Why this policy applies**\n"
            f"{why}\n\n"
            f"**Evaluated Policies (Policy Traceability)**\n"
            f"{evaluated_block}\n\n"
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
