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

    Uses missing_information to ask targeted questions:
      - "location"              → ask only for city
      - "activity"              → ask only for activity
      - "location_and_activity" → ask for both
      - indoor context          → ask for city (gym context)
      - None / fallback         → ask for both
    """
    conversation_history = state.get("conversation_history", [])
    intent = state.get("intent")
    missing_information = state.get("missing_information")
    prev_location = state.get("resolved_location") or state.get("location_text")

    logger.info(
        "clarification_response_node: requesting clarification. missing='%s'",
        missing_information,
    )

    # Detect indoor activity context (gym, yoga, etc.)
    raw_activity = getattr(intent, "raw_activity", None) if intent else None
    activity_cats = getattr(intent, "activity_categories", []) if intent else []
    is_indoor = (
        "indoor_activity" in activity_cats
        or raw_activity in {"indoor_gym", "indoor_exercise"}
    )

    # Build targeted question based on missing information
    if is_indoor and raw_activity == "indoor_gym":
        # Special gym case: gym is indoors, so weather is generally not a concern,
        # but ask for city in case the user is asking about traveling TO the gym.
        question = (
            "🤔 **A quick question about your gym plans!**\n\n"
            "Gym sessions are indoors, so weather usually won't impact your workout directly. "
            "However, if you're asking about **traveling to the gym** or doing an **outdoor workout**, "
            "I'd be happy to check conditions.\n\n"
            "Could you let me know:\n"
            "- **Where** are you located? *(city name)*\n"
            "- Are you asking about **traveling to the gym** or an **outdoor exercise** activity?"
        )

    elif missing_information == "location":
        activity_name = raw_activity or (activity_cats[0] if activity_cats else "this activity")
        if activity_name in {"indoor_gym", "indoor_exercise"}:
            question = (
                "🤔 **Which city are you in?**\n\n"
                "I can check weather conditions for traveling to the gym or any outdoor activity. "
                "Just tell me your city and I'll look it up!"
            )
        else:
            activity_display = activity_name.replace("_", " ").title() if activity_name != "general" else "your activity"
            question = (
                f"🤔 **Which city are you in?**\n\n"
                f"I'd love to check conditions for **{activity_display}** — "
                f"just tell me the city and I'll pull up the latest weather and safety guidance!"
            )

    elif missing_information == "activity":
        location_display = prev_location or "your location"
        question = (
            f"🤔 **What activity are you planning?**\n\n"
            f"I can check safety for **{location_display}** — could you tell me what you're planning?\n\n"
            "For example:\n"
            "- *Cycling, walking, running*\n"
            "- *Picnic or park visit*\n"
            "- *Commuting by scooter or car*\n"
            "- *Taking children or elderly outdoors*"
        )

    elif missing_information == "location_and_activity":
        question = (
            "🤔 **Could you tell me a bit more?**\n\n"
            "I need two things to give you accurate weather-safety guidance:\n\n"
            "- **Where** are you planning to go? *(city name)*\n"
            "- **What activity** are you planning? *(e.g. cycling, walking, picnic)*"
        )

    elif intent and intent.time_context and intent.time_context not in {"current", "today"}:
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
                "Could you tell me:\n"
                "- **Where** are you planning this activity?\n"
                "- **What** activity are you planning?"
            )

    else:
        question = (
            "🤔 **Could you clarify?**\n\n"
            "I'm not sure what you're asking about. Could you tell me:\n"
            "- **Where** are you planning to go? *(city name)*\n"
            "- **What activity** are you planning? *(e.g. cycling, walking, park visit)*"
        )

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": question}
    ]

    return {
        "final_answer": question,
        "conversation_history": updated_history,
        "needs_clarification": False,  # reset for next turn
        "missing_information": None,   # reset for next turn
    }
