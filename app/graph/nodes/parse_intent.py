"""
Intent parsing node.
Extracts structured intent from user input, resolves session continuity,
and handles out-of-scope or ambiguous requests.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.llm.intent_parser import _heuristic_parse_intent, parse_intent
from app.llm.normalizer import normalize_input

logger = logging.getLogger(__name__)

def parse_intent_node(state: BotState) -> Dict:
    """
    Parse user message into structured intent and resolve against session state.
    Implements field-level context merging:
      - New value exists -> update field
      - New value missing -> preserve previous context
    """
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])

    prev_intent = state.get("intent")
    prev_location = state.get("location_text") or state.get("resolved_location")
    prev_activity = state.get("activity")
    prev_time = state.get("requested_time")
    prev_mode = prev_intent.mode if prev_intent else None
    prev_group = prev_intent.group if prev_intent else None
    prev_categories = prev_intent.activity_categories if prev_intent else None
    has_prior_context = bool(prev_location or prev_activity or prev_intent)

    logger.info("=== CONTEXT RESOLUTION START ===")
    logger.info("USER QUERY: '%s'", user_message)
    logger.info(
        "PREVIOUS CONTEXT: location='%s', activity='%s', time='%s'",
        prev_location, prev_activity, prev_time,
    )

    # Normalize typos and informal phrasing before calling the LLM
    normalized_message, interpreted_as = normalize_input(user_message)
    if interpreted_as:
        logger.info("Input normalized: '%s' → '%s'", user_message, normalized_message)

    # Classify intent with Gemini (with heuristic fallback)
    try:
        intent = parse_intent(
            user_message=normalized_message,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        logger.warning("Gemini parsing failed (%s); trying heuristic fallback.", exc)
        intent = _heuristic_parse_intent(normalized_message)
        if intent is None:
            if has_prior_context:
                logger.info("Intent unparsed but active session context exists — retaining prior context.")
                intent = ParsedIntent(
                    activity_categories=prev_categories or ["general"],
                    mode=prev_mode,
                    group=prev_group,
                    location=prev_location,
                    time_context=prev_time or "current",
                    is_follow_up=True,
                    raw_activity=prev_activity,
                )
            else:
                logger.error("Intent parsing completely failed: %s", exc)
                return {
                    "error": f"Intent parsing failed: {exc}",
                    "error_type": "llm_failure",
                    "interpreted_as": interpreted_as,
                }

    logger.info(
        "PARSED NEW INTENT: loc=%s, cats=%s, mode=%s, time=%s, is_follow_up=%s",
        intent.location, intent.activity_categories, intent.mode, intent.time_context, intent.is_follow_up,
    )

    # Handle empty/unrecognized intents when no context is available
    if (
        not intent.activity_categories
        and not intent.location
        and not intent.time_context
        and not has_prior_context
    ):
        logger.info("Empty intent with no prior context — treating as out-of-scope.")
        updated_history = list(conversation_history) + [
            {"role": "user", "content": user_message}
        ]
        return {
            "scope_type": "irrelevant",
            "conversation_history": updated_history,
            "interpreted_as": interpreted_as,
            "error": None,
            "error_type": None,
        }

    # =========================================================================
    # FIELD-LEVEL CONTEXT MERGING:
    # 1. Location: new location replaces old; missing location preserves old.
    # 2. Activity: new activity replaces old; missing activity preserves old.
    # 3. Time: new time replaces old; missing time preserves old (or defaults to current).
    # =========================================================================

    # 1. Location merging
    location_changed = False
    if intent.location and intent.location.strip():
        merged_location = intent.location.strip()
        if prev_location and merged_location.lower() != prev_location.lower():
            location_changed = True
    elif prev_location:
        merged_location = prev_location
    else:
        merged_location = None
    intent.location = merged_location

    # 2. Activity merging
    has_new_activity = bool(intent.activity_categories or intent.mode or intent.raw_activity)
    if has_new_activity:
        merged_activity = (
            intent.mode
            or intent.raw_activity
            or (intent.activity_categories[0] if intent.activity_categories else None)
        )
        merged_categories = intent.activity_categories or ["general"]
        merged_mode = intent.mode
        merged_group = intent.group or prev_group
        merged_raw_activity = intent.raw_activity or intent.mode
    elif has_prior_context and (prev_activity or prev_categories):
        merged_activity = prev_activity
        merged_categories = (
            prev_categories
            or (["outdoor_exercise"] if prev_activity in ["walking", "cycling", "running"] else ["general"])
        )
        merged_mode = prev_mode or (
            prev_activity if prev_activity in [
                "walking", "cycling", "running", "car", "scooter", "swimming", "boating"
            ] else None
        )
        merged_group = prev_group
        merged_raw_activity = prev_intent.raw_activity if prev_intent else prev_activity
    else:
        merged_activity = None
        merged_categories = ["general"]
        merged_mode = None
        merged_group = None
        merged_raw_activity = None

    intent.activity_categories = merged_categories
    intent.mode = merged_mode
    intent.group = merged_group
    intent.raw_activity = merged_raw_activity

    # 3. Time merging
    if intent.time_context and intent.time_context.strip():
        merged_time = intent.time_context.strip()
    elif prev_time:
        merged_time = prev_time
    else:
        merged_time = "current"
    intent.time_context = merged_time

    logger.info(
        "MERGED CONTEXT: location='%s', activity='%s', time='%s'",
        merged_location, merged_activity, merged_time,
    )
    logger.info("=== CONTEXT RESOLUTION END ===")

    # Clarification Check: if after merging we STILL have no location, ask for clarification
    needs_clarification = False
    if not merged_location:
        logger.info("No location resolved from query or session memory — requesting clarification.")
        needs_clarification = True

    updated_history = list(conversation_history) + [
        {"role": "user", "content": user_message}
    ]

    ret = {
        "intent": intent,
        "activity": merged_activity or "general",
        "requested_time": merged_time,
        "needs_clarification": needs_clarification,
        "scope_type": "relevant",
        "interpreted_as": interpreted_as,
        "location_text": merged_location,
        "conversation_history": updated_history,
        "error": None,
        "error_type": None,
    }

    if location_changed:
        ret["lat"] = None
        ret["lon"] = None
        ret["resolved_location"] = None

    return ret

