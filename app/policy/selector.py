from __future__ import annotations

import logging
from typing import List

from app.policy.models import MatchResult, PolicyDecision

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"critical": 4, "high": 3, "moderate": 2, "low": 1}


def _severity_key(match: MatchResult) -> tuple:
    """Sort key prioritizing higher severity first, breaking ties with priority rank."""
    return (-_SEVERITY_RANK.get(match.severity, 0), match.priority)


def resolve_policy(matches: List[MatchResult]) -> PolicyDecision:
    """
    Select the primary SOP from a list of matched SOPs using deterministic rules:
    1. If any override SOPs fired, restrict primary candidate selection to overrides.
    2. Pick the candidate with the highest severity (critical > high > moderate > low).
    3. Break severity ties using the lowest numerical priority rank.
    4. Designate all remaining matches as secondary advisories.
    """
    if not matches:
        raise ValueError("resolve_policy called with empty matches list.")

    # Prioritize override SOPs if present
    override_matches = [m for m in matches if m.overrides]
    eligible = override_matches if override_matches else matches

    if override_matches:
        reason_prefix = (
            f"{len(override_matches)} override SOP(s) active; "
            "overrides take precedence for primary advisory."
        )
    else:
        reason_prefix = "All matched SOPs eligible."

    # Rank by severity and priority
    sorted_eligible = sorted(eligible, key=_severity_key)
    primary = sorted_eligible[0]
    secondary = [m for m in matches if m.sop_id != primary.sop_id]

    reason = (
        f"{reason_prefix} "
        f"Selected {primary.sop_id} (severity={primary.severity}, priority={primary.priority})."
    )
    logger.info("Policy resolved: primary=%s | reason: %s", primary.sop_id, reason)

    return PolicyDecision(
        primary=primary,
        secondary_matches=secondary,
        resolution_reason=reason,
    )
