"""
Node: clarification_response
Produced when intent is ambiguous and the system cannot safely resolve context.
Asks the user a targeted clarifying question instead of guessing.
No LLM call — deterministic canned clarification questions.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)


def clarification_response_node(state: BotState) -> Dict:
    """
    LangGraph node: produce a clarification question.

    Called when parse_intent sets needs_clarification=True.
    The system cannot safely resolve what the user is asking without
    more information, and must not guess.
    """
    conversation_history = state.get("conversation_history", [])
    intent = state.get("intent")
    prev_location = state.get("resolved_location") or state.get("location_text")

    logger.info("clarification_response_node: requesting clarification from user.")

    # Build a targeted question based on what we know
    if intent and intent.time_context and intent.time_context not in {"current", "today"}:
        time_ref = intent.time_context
        if prev_location:
            question = (
                f"🤔 **Could you clarify?**\n\n"
                f"You asked about **{time_ref}** — "
                f"did you mean **{time_ref}** in **{prev_location}** "
                f"for the same activity as before?\n\n"
                f"If not, please let me know the location and activity you have in mind."
            )
        else:
            question = (
                f"🤔 **Could you clarify?**\n\n"
                f"I'm not sure what location or activity you're referring to with \"{time_ref}\".\n\n"
                f"Could you tell me:\n"
                f"- **Where** are you planning this activity?\n"
                f"- **What** activity are you planning?"
            )
    else:
        question = (
            "🤔 **Could you clarify?**\n\n"
            "I'm not sure what you're asking about. Could you tell me:\n"
            "- **Where** are you planning to go?\n"
            "- **What activity** are you planning (e.g. cycling, walking, park visit)?"
        )

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": question}
    ]

    return {
        "final_answer": question,
        "conversation_history": updated_history,
        "needs_clarification": False,  # reset for next turn
    }
