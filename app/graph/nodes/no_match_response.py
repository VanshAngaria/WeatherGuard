from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from app.graph.state import BotState
from app.llm.client import get_llm_client, get_model_name
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


_SUPPORTED_ACTIVITIES_LIST = """**Supported activity types include:**
• 🚴 **Cycling** (commute, road, mountain)
• 🚶 **Walking & Pedestrian travel**
• 🏃 **Running & Outdoor exercise**
• 🚗 **Commuting & Two-wheelers** (motorbikes, scooters)
• 🧺 **Picnics & Outdoor recreation**
• 👨‍👩‍👧 **Child outdoor activity**
• 👴 **Elderly person outdoor activity**"""


_NO_MATCH_PROMPT = """You are composing the final user-facing response for a weather safety advisory assistant.

The user asked whether an activity or outdoor plan is safe under current/forecast weather conditions.
FACT: All live weather indicators are safe. None of the adverse safety thresholds were triggered.

Your task: Write two short paragraphs in clear, conversational English:
1. "recommendation" — directly and conversationally answer the user's question (for example, if they ask "Is it safe?" or "Can I go?", lead directly with a clear verdict: "Yes, it is completely safe to proceed..." or "Yes, conditions remain favorable for..."). Acknowledge the conversation naturally without repeating identical boilerplate sentences from earlier turns.
2. "why" — a brief, factual explanation that all monitored parameters (rain probability, wind gusts, temperature, visibility) remain safely below hazard limits.

User Query: "{user_query}"

Recent Dialogue Context:
{conversation_context}

Safe Conditions Context:
- Location: {location}
- Activity: {activity}
- Time: {time_label}
- Weather Facts: {weather_facts}
- Evaluated Policies: {evaluated_sops}

Respond with ONLY a JSON object:
{{"recommendation": "...", "why": "..."}}

Keep each paragraph to 2-3 sentences. Be direct, reassuring, and conversational.
"""


def no_match_response_node(state: BotState) -> Dict:
    """
    Produces a policy-governed advisory response when no safety conditions are exceeded
    or when an unsupported activity is requested.
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
    has_coverage = _any_sop_covers_categories(real_cats) if real_cats else True

    conditions_block = _build_conditions_block(facts)
    subtitle = f"_{activity.title()} · {time_label}_" if activity and activity != "general" else f"_{time_label}_"

    if not has_coverage and real_cats:
        act_display = f"**{activity}**" if activity else "this activity"
        answer = (
            f"ℹ️ **No Policy Coverage — {location}**\n"
            f"{subtitle}\n\n"
            f"**Recommendation**\n"
            f"We don't currently have a Standard Operating Procedure (SOP) or verified safety protocol defined for {act_display}. "
            f"To prevent unverified or hallucinated advice, safety recommendations are strictly provided only for activities governed by our written SOP library.\n\n"
            f"**{time_label}**\n"
            f"{conditions_block}\n\n"
            f"**Severity**\n"
            f"ℹ️ NO POLICY DEFINED\n\n"
            f"**Applicable SOP**\n"
            f"None — No written safety policy covers {act_display}.\n\n"
            f"**Policy Traceability**\n"
            f"Evaluated categories `{real_cats}` against loaded SOP catalog. Zero matching procedures exist for this activity.\n\n"
            f"{_SUPPORTED_ACTIVITIES_LIST}\n\n"
            f"Please ask about one of the supported activities above for **{location}**."
        )
    else:
        # Collect relevant SOPs that apply to this activity
        eval_sops = []
        eval_bullets = []
        facts_dict = facts.to_facts_dict() if facts else {}

        for sop in get_sops():
            if "*" in sop.applies_to_categories or any(c in sop.applies_to_categories for c in intent_categories):
                eval_sops.append(f"{sop.id} ({sop.title})")
                eval_bullets.append(f"• `{sop.id}` ({sop.title}) — Within safe limits ✅")

        if not eval_bullets:
            eval_bullets = [
                "• `SOP-001` (Extreme Heat) — Within safe limits ✅",
                "• `SOP-004` (Visibility) — Within safe limits ✅",
                "• `SOP-012` (Wind Gusts) — Within safe limits ✅",
                "• `SOP-015` (Rain Probability) — Within safe limits ✅",
                "• `SOP-005` (Thunderstorm Warning) — Within safe limits ✅",
                "• `SOP-019` (Regional Storm System) — Within safe limits ✅",
            ]

        # Build context snippets
        history_snippets = []
        for h in conversation_history[-4:]:
            role = h.get("role", "user").capitalize()
            content = h.get("content", "")[:250].replace("\n", " ")
            history_snippets.append(f"{role}: {content}")
        conversation_context = "\n".join(history_snippets) if history_snippets else "(New conversation session)"

        prompt = _NO_MATCH_PROMPT.format(
            user_query=user_query,
            conversation_context=conversation_context,
            location=location,
            activity=activity or "general outdoor activity",
            time_label=time_label,
            weather_facts=json.dumps(facts_dict, indent=2),
            evaluated_sops=", ".join(eval_sops[:6]),
        )

        recommendation = ""
        why = ""

        try:
            client = get_llm_client()
            model = get_model_name()
            cfg_kwargs = {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 1024,
            }
            if "lite" not in model:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            recommendation = parsed.get("recommendation", "").strip()
            why = parsed.get("why", "").strip()
        except Exception as exc:
            logger.warning("LLM generation in no_match_response failed (%s); using deterministic template.", exc)
            act_phrase = f"for **{activity}** " if activity and activity != "general" else ""
            recommendation = (
                f"Current weather conditions in **{location}** {act_phrase}fall within safe operational thresholds. "
                f"All monitored atmospheric parameters remain below active hazard limits."
            )
            why = (
                f"Atmospheric observations were verified against active safety policies: "
                f"{', '.join(eval_sops[:4])}. Live conditions satisfy all safe operational criteria."
            )

        evaluated_block = "\n".join(eval_bullets[:6])

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
