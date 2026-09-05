"""
Node: generate_response
Uses the LLM to compose a natural, conversational response FROM the already-determined
policy decision. The LLM receives structured verified inputs and may only compose language.

The LLM MUST NOT:
- Select a different SOP
- Modify severity
- Invent thresholds or weather values
- Override the policy engine decision

The final response always uses the standardized format:
🌦️ Weather Advisory — LOCATION

Recommendation
<LLM prose based on SOP advice_template>

Current Conditions
• field: actual_value  (only fields available in WeatherFacts)

Severity
<from selected_sop>

Applicable SOP
<SOP-ID> — <SOP title>

Why this policy applies
<LLM prose connecting conditions to SOP>
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from google import genai
from google.genai import types

from app.graph.state import BotState
from app.llm.client import get_llm_client, get_model_name

logger = logging.getLogger(__name__)

# Severity → display string
_SEVERITY_LABEL = {
    "critical": "🚨 CRITICAL",
    "high": "⚠️ HIGH",
    "moderate": "🟡 MODERATE",
    "low": "✅ LOW",
}

# Severity → leading emoji for header
_SEVERITY_EMOJI = {
    "critical": "🚨",
    "high": "⚠️",
    "moderate": "⚠️",
    "low": "🌦️",
}


def _build_conditions_block(facts) -> str:
    """Build the 'Current Conditions' bullet list from only non-None WeatherFacts."""
    lines = []
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


def _build_structured_response(
    location: str,
    severity: str,
    sop_id: str,
    sop_title: str,
    recommendation: str,
    why: str,
    conditions_block: str,
    secondary_ids: Optional[str] = None,
) -> str:
    """Assemble the standardized response format."""
    emoji = _SEVERITY_EMOJI.get(severity, "🌦️")
    severity_label = _SEVERITY_LABEL.get(severity, severity.upper())

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
        f"`{sop_id}` — {sop_title}",
        "",
        "**Why this policy applies**",
        why.strip(),
    ]

    if secondary_ids:
        parts += [
            "",
            "---",
            f"*Additional policies also triggered: {secondary_ids}. "
            "The primary response above reflects the highest-priority policy.*",
        ]

    return "\n".join(parts)


_PROMPT_TEMPLATE = """You are composing the final user-facing response for a weather safety advisory.

You have been given VERIFIED, FACTUAL inputs. You MUST NOT:
- Change the severity level
- Change the applicable SOP
- Invent weather values
- Override the safety recommendation
- Add thresholds not present in the SOP

Your task: Write two short paragraphs in clear, conversational English:
1. "recommendation" — what the user should do or avoid (based on the SOP advice below)
2. "why" — a short explanation of why this SOP applies to the current conditions

Inputs:
{inputs_json}

Respond with ONLY a JSON object:
{{"recommendation": "...", "why": "..."}}

Keep each paragraph to 2-3 sentences. Be direct. Lead with the safety verdict for HIGH/CRITICAL severity.
"""


def generate_response_node(state: BotState) -> Dict:
    """
    LangGraph node: compose final natural-language response using LLM.

    The LLM receives structured verified inputs and may only compose language.
    Policy decision, severity, and SOP selection are already fixed in state.
    """
    decision = state.get("policy_decision")
    facts = state.get("weather_facts")
    location = state.get("resolved_location", "your location")
    conversation_history = state.get("conversation_history", [])
    activity = state.get("activity", "this activity")
    requested_time = state.get("requested_time", "current")

    if decision is None or facts is None:
        logger.error("generate_response_node: policy_decision or weather_facts missing.")
        return {
            "final_answer": (
                "I encountered an internal error composing the response. Please try again."
            )
        }

    primary = decision.primary
    conditions_block = _build_conditions_block(facts)
    secondary_ids = (
        ", ".join(m.sop_id for m in decision.secondary_matches)
        if decision.secondary_matches else None
    )

    # Build structured inputs for the LLM (only verified facts)
    facts_dict = {k: v for k, v in facts.to_facts_dict().items() if v is not None}
    llm_inputs = {
        "location": location,
        "activity": activity,
        "requested_time": requested_time,
        "sop_id": primary.sop_id,
        "sop_title": primary.sop_title,
        "severity": primary.severity,
        "sop_advice": primary.advice_template.strip(),
        "matched_conditions": primary.matched_conditions,
        "weather_facts": facts_dict,
    }

    prompt = _PROMPT_TEMPLATE.format(inputs_json=json.dumps(llm_inputs, indent=2))

    recommendation = ""
    why = ""

    try:
        client = get_llm_client()
        model = get_model_name()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
        recommendation = parsed.get("recommendation", "").strip()
        why = parsed.get("why", "").strip()
        logger.info("generate_response_node: LLM narrative generated for %s", primary.sop_id)
    except Exception as exc:
        logger.warning("LLM narrative generation failed (%s); falling back to template.", exc)
        # Graceful fallback: use template text directly
        from app.graph.nodes.compose_answer import _format_template
        facts_flat = facts.to_facts_dict()
        recommendation = _format_template(primary.advice_template.strip(), facts_flat)
        why = f"Live conditions met the threshold defined in {primary.sop_id}."

    final_answer = _build_structured_response(
        location=location,
        severity=primary.severity,
        sop_id=primary.sop_id,
        sop_title=primary.sop_title,
        recommendation=recommendation,
        why=why,
        conditions_block=conditions_block,
        secondary_ids=secondary_ids,
    )

    # Prepend "Interpreted as" indicator if input was normalized
    interpreted_as = state.get("interpreted_as")
    if interpreted_as:
        prefix = f"_💬 Interpreted as: \"{interpreted_as}\"_\n\n"
        final_answer = prefix + final_answer

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": final_answer}
    ]

    logger.info("generate_response_node: final answer composed for %s", primary.sop_id)

    return {
        "final_answer": final_answer,
        "conversation_history": updated_history,
    }
