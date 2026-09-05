"""
Intent parsing node.
Extracts structured intent from user input, resolves session continuity,
and handles out-of-scope or ambiguous requests.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.llm.intent_parser import parse_intent
from app.llm.normalizer import normalize_input

logger = logging.getLogger(__name__)

# Heuristic filter for queries completely outside the weather-safety domain
_IRRELEVANT_KEYWORDS = {
    "python", "javascript", "code", "program", "algorithm", "database",
    "capital of", "what is the", "explain", "define", "history of",
    "write a", "give me a", "tell me about", "who is", "when was",
    "recipe", "cook", "translate", "convert", "calculate", "math",
    "joke", "story", "poem", "song", "game",
}


def _is_likely_irrelevant(message: str) -> bool:
    """Quick check for clearly off-topic queries."""
    lower = message.lower()
    return any(kw in lower for kw in _IRRELEVANT_KEYWORDS)


def parse_intent_node(state: BotState) -> Dict:
    """
    Parse user message into structured intent and resolve against session state.
    """
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])

    logger.info("parse_intent_node: message='%s'", user_message)

    # Fast path for obvious out-of-domain queries
    if _is_likely_irrelevant(user_message) and not state.get("intent"):
        logger.info("Heuristic: likely irrelevant query — routing to scope_response.")
        updated_history = list(conversation_history) + [
            {"role": "user", "content": user_message}
        ]
        return {
            "scope_type": "irrelevant",
            "conversation_history": updated_history,
            "interpreted_as": None,
            "error": None,
            "error_type": None,
        }

    # Normalize typos and informal phrasing before calling the LLM
    normalized_message, interpreted_as = normalize_input(user_message)
    if interpreted_as:
        logger.info("Input normalized: '%s' → '%s'", user_message, normalized_message)

    # Classify intent with Gemini
    try:
        intent = parse_intent(
            user_message=normalized_message,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        logger.error("Intent parsing failed: %s", exc)
        return {
            "error": f"Intent parsing failed: {exc}",
            "error_type": "llm_failure",
            "interpreted_as": interpreted_as,
        }

    # Handle empty/unrecognized intents as out of scope
    scope_type = "relevant"
    if (
        not intent.activity_categories
        and not intent.location
        and not intent.is_follow_up
        and not state.get("intent")
    ):
        scope_type = "irrelevant"
        logger.info("LLM returned empty intent with no prior context — treating as out-of-scope.")
        updated_history = list(conversation_history) + [
            {"role": "user", "content": user_message}
        ]
        return {
            "scope_type": scope_type,
            "conversation_history": updated_history,
            "interpreted_as": interpreted_as,
            "error": None,
            "error_type": None,
        }

    # Retain location and activity across conversation turns for follow-ups
    prev_intent = state.get("intent")

    if intent.is_follow_up:
        logger.info("Follow-up query detected — merging previous session memory.")
        if not intent.location:
            intent.location = state.get("location_text") or state.get("resolved_location")

        if not intent.activity_categories and prev_intent:
            intent.activity_categories = prev_intent.activity_categories
            if not intent.mode:
                intent.mode = prev_intent.mode
            if not intent.group:
                intent.group = prev_intent.group

    # When user provides only a location, retain the prior activity
    if not intent.activity_categories and not intent.is_follow_up:
        if prev_intent and prev_intent.activity_categories:
            logger.info("Carrying over prior activity from session: %s", prev_intent.activity_categories)
            intent.activity_categories = prev_intent.activity_categories
            if not intent.mode:
                intent.mode = prev_intent.mode
            if not intent.group:
                intent.group = prev_intent.group
        else:
            intent.activity_categories = ["general"]

    # Ask for clarification if a follow-up specifies time without a known location
    needs_clarification = False
    if intent.is_follow_up and intent.time_context not in {"current", "today", None}:
        resolved_location = (
            intent.location
            or state.get("location_text")
            or state.get("resolved_location")
        )
        if not resolved_location:
            logger.info("Ambiguous follow-up without location — requesting clarification.")
            needs_clarification = True

    # Assemble structured state fields
    activity = (
        intent.raw_activity
        or intent.mode
        or (intent.activity_categories[0] if intent.activity_categories else None)
    )
    requested_time = intent.time_context
    location_text = intent.location or state.get("location_text")

    updated_history = list(conversation_history) + [
        {"role": "user", "content": user_message}
    ]

    return {
        "intent": intent,
        "activity": activity,
        "requested_time": requested_time,
        "needs_clarification": needs_clarification,
        "scope_type": scope_type,
        "interpreted_as": interpreted_as,
        "location_text": location_text,
        "conversation_history": updated_history,
        "error": None,
        "error_type": None,
    }
