"""
Node: parse_intent
Classifies user message into structured intent using the LLM.
Merges intent with session memory to support follow-up queries.

Steps:
  1. Normalize input (typo correction, informal expansion)
  2. Call LLM intent classifier
  3. Detect scope: relevant / irrelevant / ambiguous
  4. Resolve follow-up context from session memory
  5. Handle location-only messages
  6. Detect ambiguous context needing clarification
  7. Populate explicit structured state fields
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.llm.intent_parser import parse_intent
from app.llm.normalizer import normalize_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Irrelevant query detection heuristics
# These supplement the LLM — if the LLM returns empty categories AND
# no location AND no follow-up flag, we treat it as out-of-scope.
# ---------------------------------------------------------------------------

_IRRELEVANT_KEYWORDS = {
    "python", "javascript", "code", "program", "algorithm", "database",
    "capital of", "what is the", "explain", "define", "history of",
    "write a", "give me a", "tell me about", "who is", "when was",
    "recipe", "cook", "translate", "convert", "calculate", "math",
    "joke", "story", "poem", "song", "game",
}


def _is_likely_irrelevant(message: str) -> bool:
    """
    Heuristic check for obviously irrelevant messages.
    Returns True only for high-confidence irrelevant queries.
    """
    lower = message.lower()
    return any(kw in lower for kw in _IRRELEVANT_KEYWORDS)


def parse_intent_node(state: BotState) -> Dict:
    """
    LangGraph node: parse user intent with full context resolution.

    Returns updated state fields including:
    - intent, activity, requested_time, needs_clarification
    - scope_type, interpreted_as
    - location_text, conversation_history
    - error / error_type on failure
    """
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])

    logger.info("parse_intent_node: message='%s'", user_message)

    # -------------------------------------------------------------------------
    # STEP 1: Quick irrelevance heuristic (no LLM call needed)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 2: Input normalization (typo correction + informal expansion)
    # -------------------------------------------------------------------------
    normalized_message, interpreted_as = normalize_input(user_message)
    if interpreted_as:
        logger.info("Input normalized: '%s' → '%s'", user_message, normalized_message)

    # -------------------------------------------------------------------------
    # STEP 3: LLM intent classification
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 4: Scope detection
    # If LLM returned empty categories + no location + no follow-up → irrelevant
    # -------------------------------------------------------------------------
    scope_type = "relevant"
    if (
        not intent.activity_categories
        and not intent.location
        and not intent.is_follow_up
        and not state.get("intent")  # no previous context to fall back on
    ):
        scope_type = "irrelevant"
        logger.info("LLM returned empty intent with no context — treating as irrelevant.")
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

    # -------------------------------------------------------------------------
    # STEP 5: Follow-up resolution
    # If is_follow_up=True, merge missing fields from session state.
    # -------------------------------------------------------------------------
    prev_intent = state.get("intent")

    if intent.is_follow_up:
        logger.info("Follow-up detected — merging with session memory.")

        if not intent.location:
            intent.location = state.get("location_text") or state.get("resolved_location")

        if not intent.activity_categories and prev_intent:
            intent.activity_categories = prev_intent.activity_categories
            if not intent.mode:
                intent.mode = prev_intent.mode
            if not intent.group:
                intent.group = prev_intent.group

    # -------------------------------------------------------------------------
    # STEP 6: Location-only / no-activity message
    # If user sent just a city name with no new activity, inherit previous activity.
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
            intent.activity_categories = ["general"]
            logger.info("No activity context; defaulting to 'general' category.")

    # -------------------------------------------------------------------------
    # STEP 7: Ambiguity detection
    # If follow-up with time-only and no resolvable location → ask clarification.
    # -------------------------------------------------------------------------
    needs_clarification = False
    if intent.is_follow_up and intent.time_context not in {"current", "today", None}:
        resolved_location = (
            intent.location
            or state.get("location_text")
            or state.get("resolved_location")
        )
        if not resolved_location:
            logger.info("Ambiguous follow-up — no resolvable location; requesting clarification.")
            needs_clarification = True

    # -------------------------------------------------------------------------
    # STEP 8: Populate explicit structured state fields
    # -------------------------------------------------------------------------
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
