"""
Conflict-resolution / policy selector.
Deterministically selects the primary SOP when multiple SOPs match.

Resolution rules (in order):
  1. If any override SOPs match, only those are eligible to become primary.
  2. Among eligible SOPs, higher severity wins.
     (critical > high > moderate > low)
  3. On severity tie, lower numeric priority wins.
  4. All remaining matches become secondary_matches.

No LLM call at any point in this module.
"""

from __future__ import annotations

import logging
from typing import List

from app.policy.models import MatchResult, PolicyDecision

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"critical": 4, "high": 3, "moderate": 2, "low": 1}


def _severity_key(match: MatchResult) -> tuple:
    """Sort key: descending severity rank, ascending priority number."""
    return (-_SEVERITY_RANK.get(match.severity, 0), match.priority)


def resolve_policy(matches: List[MatchResult]) -> PolicyDecision:
    """
    Select the primary SOP from a list of matched SOPs.

    Args:
        matches: Non-empty list of MatchResult instances.

    Returns:
        PolicyDecision with a primary SOP and list of secondary matches.

    Raises:
        ValueError: If matches is empty (caller should guard against this).
    """
    if not matches:
        raise ValueError("resolve_policy called with empty matches list.")

    # --- Step 1: override filter ---
    override_matches = [m for m in matches if m.overrides]
    eligible = override_matches if override_matches else matches

    if override_matches:
        reason_prefix = (
            f"{len(override_matches)} override SOP(s) present; "
            "only override SOPs are eligible for primary selection."
        )
    else:
        reason_prefix = "No override SOPs; all matched SOPs are eligible."

    # --- Steps 2 & 3: severity then priority ---
    sorted_eligible = sorted(eligible, key=_severity_key)
    primary = sorted_eligible[0]

    # --- Step 4: secondary matches (everything that is not primary) ---
    secondary = [m for m in matches if m.sop_id != primary.sop_id]

    reason = (
        f"{reason_prefix} "
        f"Primary selected: {primary.sop_id} "
        f"(severity={primary.severity}, priority={primary.priority})."
    )
    logger.info("Policy resolved: primary=%s | reason: %s", primary.sop_id, reason)

    return PolicyDecision(
        primary=primary,
        secondary_matches=secondary,
        resolution_reason=reason,
    )
