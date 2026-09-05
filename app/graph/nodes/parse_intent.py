"""
Node: parse_intent
Classifies user message into structured intent using the LLM.
Merges intent with session memory to support follow-up queries.
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

    - Calls the LLM intent parser with conversation history for context.
    - Merges follow-up intents with previous session state (location, mode, group).
    - Returns updated state fields.
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

    # --- Follow-up resolution ---
    # If this is a follow-up query, inherit previous location/mode/group from state.
    if intent.is_follow_up:
        logger.info("Follow-up query detected — merging with session memory.")

        # Inherit location from session if not provided in current turn
        if not intent.location:
            intent.location = state.get("location_text") or state.get("resolved_location")

        # Inherit categories if empty
        if not intent.activity_categories:
            prev_intent = state.get("intent")
            if prev_intent:
                intent.activity_categories = prev_intent.activity_categories
                if not intent.mode:
                    intent.mode = prev_intent.mode
                if not intent.group:
                    intent.group = prev_intent.group

    # Determine location text for next node
    location_text = intent.location or state.get("location_text")

    # Append current user message to conversation history
    updated_history = list(conversation_history) + [
        {"role": "user", "content": user_message}
    ]

    return {
        "intent": intent,
        "location_text": location_text,
        "conversation_history": updated_history,
        "error": None,
        "error_type": None,
    }
