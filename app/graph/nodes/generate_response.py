from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from google import genai
from google.genai import types

from app.graph.state import BotState
from app.llm.client import get_llm_client, get_model_name

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


def _build_conditions_block(facts) -> str:
    """Build formatted bullet points from non-empty weather facts."""
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


def _build_traceability_block(primary, activity: Optional[str] = None, why_text: Optional[str] = None) -> str:
    """Build deterministic traceable explanation of why this policy applied."""
    lines = []
    if activity and activity != "general":
        lines.append(f"• **Activity**: {activity}")
    
    traces = getattr(primary, "condition_traces", [])
    if traces:
        for t in traces:
            field_name = t.get("field", "").replace("_", " ").title()
            obs = t.get("observed")
            op = t.get("operator")
            thresh = t.get("threshold")
            lines.append(f"• **{field_name}**: {obs} (Policy threshold: {op} {thresh})")
    elif primary.matched_conditions:
        for f, val in primary.matched_conditions.items():
            field_name = f.replace("_", " ").title()
            lines.append(f"• **{field_name}**: {val}")

    lines.append(f"• **Severity Level**: {primary.severity.upper()}")
    
    trace_summary = "\n".join(lines)
    if why_text and why_text.strip():
        return f"{why_text.strip()}\n\n{trace_summary}"
    return trace_summary


def _build_structured_response(
    location: str,
    severity: str,
    sop_id: str,
    sop_title: str,
    recommendation: str,
    why: str,
    conditions_block: str,
    time_label: str = "Current Conditions",
    activity: Optional[str] = None,
    secondary_ids: Optional[str] = None,
) -> str:
    """Format the standardized user-facing advisory report."""
    emoji = _SEVERITY_EMOJI.get(severity, "🌦️")
    severity_label = _SEVERITY_LABEL.get(severity, severity.upper())
    subtitle = f"_{activity.title()} · {time_label}_" if activity and activity != "general" else f"_{time_label}_"

    parts = [
        f"{emoji} **Weather Advisory — {location}**",
        subtitle,
        "",
        "**Recommendation**",
        recommendation.strip(),
        "",
        f"**{time_label}**",
        conditions_block,
        "",
        "**Severity**",
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


_PROMPT_TEMPLATE = """You are composing the final user-facing response for a weather safety advisory assistant.

You have been given VERIFIED, FACTUAL inputs. You MUST NOT:
- Change the severity level
- Change the applicable SOP
- Invent weather values
- Override the safety recommendation
- Add thresholds not present in the SOP

Your task: Write two short paragraphs in clear, conversational English:
1. "recommendation" — directly and conversationally answer the user's question (for example, if they ask "Is it safe?" or "Can I go?", lead directly with a clear verdict: "No, cycling is not recommended this evening due to..." or "Yes, it is safe to proceed..."). Acknowledge the conversation naturally without repeating identical boilerplate sentences from earlier turns.
2. "why" — a short explanation of why this SOP applies to the live atmospheric conditions.

User Query: "{user_query}"

Recent Dialogue Context:
{conversation_context}

Factual Inputs:
{inputs_json}

Respond with ONLY a JSON object:
{{"recommendation": "...", "why": "..."}}

Keep each paragraph to 2-3 sentences. Be direct and conversational. Lead with the safety verdict for HIGH/CRITICAL severity.
"""



def generate_response_node(state: BotState) -> Dict:
    """
    Synthesizes conversational narrative grounded in the policy decision and weather facts.
    Falls back gracefully to template composition if LLM response is unavailable.
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
            "final_answer": "I encountered an internal error composing the response. Please try again."
        }

    primary = decision.primary
    time_label = getattr(facts, "time_label", None) or "Current Conditions"
    conditions_block = _build_conditions_block(facts)
    secondary_ids = (
        ", ".join(m.sop_id for m in decision.secondary_matches)
        if decision.secondary_matches else None
    )

    facts_dict = {k: v for k, v in facts.to_facts_dict().items() if v is not None}
    llm_inputs = {
        "location": location,
        "activity": activity,
        "time_context": time_label,
        "sop_id": primary.sop_id,
        "sop_title": primary.sop_title,
        "severity": primary.severity,
        "sop_advice": primary.advice_template.strip(),
        "matched_conditions": primary.matched_conditions,
        "weather_facts": facts_dict,
    }

    user_query = state.get("user_message", "")
    history_snippets = []
    for h in conversation_history[-4:]:
        role = h.get("role", "user").capitalize()
        # truncate content snippet
        content = h.get("content", "")[:250].replace("\n", " ")
        history_snippets.append(f"{role}: {content}")
    conversation_context = "\n".join(history_snippets) if history_snippets else "(New conversation session)"

    prompt = _PROMPT_TEMPLATE.format(
        user_query=user_query,
        conversation_context=conversation_context,
        inputs_json=json.dumps(llm_inputs, indent=2),
    )
    recommendation = ""
    why = ""

    try:
        client = get_llm_client()
        model = get_model_name()
        cfg_kwargs = {
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "max_output_tokens": 2048,
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
        logger.warning("LLM response generation failed (%s); using deterministic template.", exc)
        from app.graph.nodes.compose_answer import _format_template
        facts_flat = facts.to_facts_dict()
        recommendation = _format_template(primary.advice_template.strip(), facts_flat)
    traceable_why = _build_traceability_block(primary, activity=activity, why_text=why)

    final_answer = _build_structured_response(
        location=location,
        severity=primary.severity,
        sop_id=primary.sop_id,
        sop_title=primary.sop_title,
        recommendation=recommendation,
        why=traceable_why,
        conditions_block=conditions_block,
        time_label=time_label,
        activity=activity,
        secondary_ids=secondary_ids,
    )


    interpreted_as = state.get("interpreted_as")
    if interpreted_as:
        prefix = f"_💬 Interpreted as: \"{interpreted_as}\"_\n\n"
        final_answer = prefix + final_answer

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": final_answer}
    ]

    return {
        "final_answer": final_answer,
        "conversation_history": updated_history,
    }
