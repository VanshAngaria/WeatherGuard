"""
Node: resolve_policy
Applies deterministic conflict resolution to select the primary SOP
when multiple SOPs match. Resolution: override > severity > priority.
No LLM call.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.graph.state import BotState
from app.policy.selector import resolve_policy

logger = logging.getLogger(__name__)


def resolve_policy_node(state: BotState) -> Dict:
    """
    LangGraph node: select primary SOP from matched SOPs.

    - Reads sop_matches from state.
    - Applies deterministic conflict resolution.
    - Returns policy_decision.
    """
    matches = state.get("sop_matches")

    if not matches:
        # This node should only be called when matches is non-empty
        # (the graph routes empty matches to no_match_response).
        logger.warning("resolve_policy_node called with empty matches — unexpected.")
        return {}

    logger.info("resolve_policy_node: resolving %d matches.", len(matches))

    decision = resolve_policy(matches)

    logger.info(
        "Policy resolved: primary=%s | secondary=%s | reason=%s",
        decision.primary.sop_id,
        [m.sop_id for m in decision.secondary_matches],
        decision.resolution_reason,
    )

    return {"policy_decision": decision}
