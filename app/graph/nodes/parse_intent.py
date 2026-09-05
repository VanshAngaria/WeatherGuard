"""
Node: parse_intent
Classifies user message into structured intent using the LLM.
Merges intent with session memory to support follow-up queries.
Detects ambiguous context and sets needs_clarification flag.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.llm.intent_parser import parse_intent

logger = logging.getLogger(__name__)


def parse_intent_node(state: BotState) -> Dict:
    """
    LangGraph node: parse user intent.

    Responsibilities:
    - Call LLM intent parser with conversation history for context.
    - Merge follow-up intents with previous session state.
    - Populate explicit structured state fields: activity, requested_time.
    - Set needs_clarification=True when context cannot be resolved.

    Context resolution priority:
    - New intent fields take precedence over session memory.
    - Session memory fills in missing fields from new intent.
    """
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])

    logger.info("parse_intent_node: message='%s'", user_message)

    try:
        intent = parse_intent(
            user_message=user_message,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        logger.error("Intent parsing failed: %s", exc)
        return {
            "error": f"Intent parsing failed: {exc}",
            "error_type": "llm_failure",
        }

    prev_intent = state.get("intent")

    # -------------------------------------------------------------------------
    # STEP 1: Follow-up resolution
    # If the LLM detected is_follow_up=True, merge missing fields from session.
    # -------------------------------------------------------------------------
    if intent.is_follow_up:
        logger.info("Follow-up query detected — merging with session memory.")

        # Location: inherit from session if not provided in current turn
        if not intent.location:
            intent.location = state.get("location_text") or state.get("resolved_location")

        # Activity categories: inherit if empty
        if not intent.activity_categories and prev_intent:
            intent.activity_categories = prev_intent.activity_categories
            # Only inherit mode/group if the user didn't specify a new one
            if not intent.mode:
                intent.mode = prev_intent.mode
            if not intent.group:
                intent.group = prev_intent.group

    # -------------------------------------------------------------------------
    # STEP 2: Location-only / no-activity message
    # If user sent a city name with no activity (e.g. "Roorkee"),
    # inherit previous activity from session. Default to "general" otherwise.
    # -------------------------------------------------------------------------
    if not intent.activity_categories and not intent.is_follow_up:
        if prev_intent and prev_intent.activity_categories:
            logger.info(
                "No activity in current message — inheriting from session: %s",
                prev_intent.activity_categories,
            )
            intent.activity_categories = prev_intent.activity_categories
            if not intent.mode:
                intent.mode = prev_intent.mode
            if not intent.group:
                intent.group = prev_intent.group
        else:
            # No prior context — default to general weather check
            intent.activity_categories = ["general"]
            logger.info("No activity context; defaulting to 'general' category.")

    # -------------------------------------------------------------------------
    # STEP 3: Ambiguity detection
    # If this is a follow-up that changes ONLY the time ("later", "eventually")
    # and there is no resolvable location in state, ask for clarification.
    # -------------------------------------------------------------------------
    needs_clarification = False
    if intent.is_follow_up and intent.time_context in {"evening", "night", "morning", "afternoon"}:
        # Check: do we have a resolvable location from this turn or session?
        resolved_location = (
            intent.location
            or state.get("location_text")
            or state.get("resolved_location")
        )
        if not resolved_location:
            logger.info("Ambiguous follow-up — no location resolvable; requesting clarification.")
            needs_clarification = True

    # -------------------------------------------------------------------------
    # STEP 4: Populate explicit structured fields
    # -------------------------------------------------------------------------
    # activity: prefer raw_activity from LLM, fallback to mode, then first category
    activity = (
        intent.raw_activity
        or intent.mode
        or (intent.activity_categories[0] if intent.activity_categories else None)
    )

    requested_time = intent.time_context

    # Determine location text for next node
    location_text = intent.location or state.get("location_text")

    # Append current user message to conversation history
    updated_history = list(conversation_history) + [
        {"role": "user", "content": user_message}
    ]

    return {
        "intent": intent,
        "activity": activity,
        "requested_time": requested_time,
        "needs_clarification": needs_clarification,
        "location_text": location_text,
        "conversation_history": updated_history,
        "error": None,
        "error_type": None,
    }
