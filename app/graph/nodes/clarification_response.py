"""
Node: clarification_response
Produced when intent is ambiguous and the system cannot safely resolve context.
Asks the user a targeted clarifying question instead of guessing.

No LLM call — deterministic, config-driven clarification questions loaded from
policies/messages.yaml.  To change phrasing, edit ONLY that file.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import yaml

from app.graph.state import BotState

logger = logging.getLogger(__name__)

_MESSAGES_PATH = Path(__file__).resolve().parents[4] / "policies" / "messages.yaml"


@lru_cache(maxsize=1)
def _load_templates() -> Dict:
    """Load clarification message templates from policies/messages.yaml (cached)."""
    if not _MESSAGES_PATH.exists():
        logger.warning("messages.yaml not found at %s — using inline defaults.", _MESSAGES_PATH)
        return {}
    with open(_MESSAGES_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return raw.get("clarification_messages", {})


def _render(key: str, **kwargs) -> str:
    """
    Render a clarification message from the template catalog.

    Falls back to a generic question if the key is missing.
    Template variables in {braces} are substituted from **kwargs.
    """
    templates = _load_templates()
    entry = templates.get(key) or templates.get("generic_fallback", {})

    title = entry.get("title", "🤔 **Could you clarify?**")
    body = entry.get("body", "Could you tell me more about what you need?")

    # Substitute template variables safely (missing keys left as-is)
    try:
        body = body.format(**kwargs)
    except KeyError:
        pass  # leave unresolved placeholders intact rather than crashing

    return f"{title}\n\n{body}"


def _is_indoor_category(activity_categories: list) -> bool:
    """Return True if the intent is categorised as an indoor activity by the policy engine."""
    return "indoor_activity" in (activity_categories or [])


def clarification_response_node(state: BotState) -> Dict:
    """
    LangGraph node: produce a targeted clarification question.

    Called when parse_intent sets needs_clarification=True.
    The system cannot safely resolve what the user is asking without
    more information, and must not guess.

    All message text is loaded from policies/messages.yaml — no hardcoded strings.
    """
    conversation_history = state.get("conversation_history", [])
    intent = state.get("intent")
    missing_information = state.get("missing_information")
    prev_location = state.get("resolved_location") or state.get("location_text")

    logger.info(
        "clarification_response_node: requesting clarification. missing='%s'",
        missing_information,
    )

    # Extract activity context from intent (no hardcoded activity names)
    raw_activity = getattr(intent, "raw_activity", None) if intent else None
    activity_cats = getattr(intent, "activity_categories", []) if intent else []
    time_ctx = getattr(intent, "time_context", None) if intent else None

    # Build a human-friendly activity display name from the raw value
    activity_display = (
        raw_activity.replace("_", " ").title()
        if raw_activity and raw_activity not in ("general", "weather", "general_weather")
        else (activity_cats[0].replace("_", " ").title() if activity_cats and activity_cats != ["general"] else "your activity")
    )

    # =========================================================================
    # Determine the clarification type and render from config templates
    # =========================================================================

    if _is_indoor_category(activity_cats):
        # Indoor activity: ask for city and clarify if travel vs. indoor
        question = _render("indoor_activity")

    elif missing_information == "location":
        question = _render("missing_location", activity=activity_display)

    elif missing_information == "activity":
        location_display = prev_location or "your location"
        question = _render("missing_activity", location=location_display)

    elif missing_information == "location_and_activity":
        question = _render("missing_location_and_activity")

    elif time_ctx and time_ctx not in {"current", "today"}:
        if prev_location:
            question = _render(
                "ambiguous_time_with_location",
                time_ref=time_ctx,
                location=prev_location,
            )
        else:
            question = _render("ambiguous_time_no_location", time_ref=time_ctx)

    else:
        question = _render("generic_fallback")

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": question}
    ]

    return {
        "final_answer": question,
        "conversation_history": updated_history,
        # Keep needs_clarification=True so callers can inspect that a clarification
        # was produced. The parse_intent node will reset it on the NEXT user turn.
        "needs_clarification": True,
        "missing_information": missing_information,
    }
