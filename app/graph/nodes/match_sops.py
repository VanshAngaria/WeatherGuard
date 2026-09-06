"""
Node: match_sops
Runs the deterministic SOP matcher against WeatherFacts + intent context.
No LLM call. Returns every SOP whose condition is satisfied.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.policy.matcher import match_sops

logger = logging.getLogger(__name__)


def match_sops_node(state: BotState) -> Dict:
    """
    LangGraph node: run SOP matching.

    - Reads WeatherFacts and intent from state.
    - Calls the deterministic match_sops function.
    - Returns sop_matches (may be empty list, which routes to no_match_response).
    """
    facts = state.get("weather_facts")
    intent = state.get("intent")

    if facts is None:
        logger.error("match_sops_node: WeatherFacts missing.")
        return {
            "error": "Cannot match SOPs without weather facts.",
            "error_type": "weather_failure",
        }

    if intent is None:
        logger.error("match_sops_node: Intent missing.")
        return {
            "error": "Cannot match SOPs without parsed intent.",
            "error_type": "llm_failure",
        }

    categories = intent.activity_categories or ["general"]
    mode = intent.mode
    group = intent.group

    logger.info(
        "match_sops_node: categories=%s mode=%s group=%s",
        categories, mode, group,
    )

    facts_dict = facts.to_facts_dict()
    matched = match_sops(
        facts=facts_dict,
        intent_categories=categories,
        mode=mode,
        group=group,
    )

    logger.info("MATCHED SOP: %s", [m.sop_id for m in matched])
    return {"sop_matches": matched}
