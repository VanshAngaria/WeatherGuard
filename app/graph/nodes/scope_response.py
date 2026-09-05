"""
Node: scope_response
Handles out-of-scope or irrelevant queries politely.
No LLM call, no weather fetch, no SOP evaluation.

Called when parse_intent detects scope_type = "irrelevant".
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState

logger = logging.getLogger(__name__)

_SCOPE_MESSAGE = (
    "🌦️ **Weather-Advisory Support Bot**\n\n"
    "I'm focused on **outdoor activity safety** — I can help you decide whether "
    "it's safe to cycle, walk, run, commute, or enjoy outdoor activities "
    "based on live weather and safety policies.\n\n"
    "I'm not a general-purpose assistant, so I can't answer unrelated questions.\n\n"
    "**Try asking something like:**\n"
    "- *\"Is it safe to cycle in Bhopal today?\"*\n"
    "- *\"Should I take my child to the park in Delhi?\"*\n"
    "- *\"Is today a good day for a picnic in Chandigarh?\"*\n"
    "- *\"What about this evening?\"* ← after a previous question"
)


def scope_response_node(state: BotState) -> Dict:
    """
    LangGraph node: respond to out-of-scope queries.

    Produces a polite redirect explaining the bot's purpose
    without answering the unrelated question.
    """
    conversation_history = state.get("conversation_history", [])

    logger.info("scope_response_node: out-of-scope query — returning scope message.")

    updated_history = list(conversation_history) + [
        {"role": "assistant", "content": _SCOPE_MESSAGE}
    ]

    return {
        "final_answer": _SCOPE_MESSAGE,
        "conversation_history": updated_history,
        "scope_type": "irrelevant",
    }
