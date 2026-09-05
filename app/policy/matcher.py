from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.policy.evaluator import compute_score, get_matched_fields, sop_matches
from app.policy.loader import get_sops
from app.policy.models import MatchResult, SOP

logger = logging.getLogger(__name__)

CATEGORY_ALIASES: Dict[str, List[str]] = {
    "outdoor_exercise": ["outdoor_exercise"],
    "outdoor_recreation": ["outdoor_recreation"],
    "travel": ["travel"],
    "vulnerable_groups": ["vulnerable_groups"],
    "general": ["general"],
    "water_activities": ["water_activities"],
}


def _category_applies(sop: SOP, intent_categories: List[str]) -> bool:
    """Return True if this SOP is applicable for at least one of the intent categories."""
    if "*" in sop.applies_to_categories:
        return True
    for cat in intent_categories:
        if cat in sop.applies_to_categories or cat == sop.category:
            return True
    return False


def match_sops(
    facts: Dict[str, Optional[Any]],
    intent_categories: List[str],
    mode: Optional[str] = None,
    group: Optional[str] = None,
    _sops_override: Optional[List] = None,
) -> List[MatchResult]:
    """
    Evaluate all loaded SOPs against current weather facts and intent context.
    Returns every matching SOP as a MatchResult.
    """
    sops = _sops_override if _sops_override is not None else get_sops()
    results: List[MatchResult] = []

    # Augment facts with contextual mode and group
    augmented_facts = dict(facts)
    if mode:
        augmented_facts["mode"] = mode
    if group:
        augmented_facts["group"] = group

    for sop in sops:
        if not _category_applies(sop, intent_categories):
            continue

        if not sop_matches(sop, augmented_facts):
            continue

        matched = get_matched_fields(sop, augmented_facts)

        score: Optional[float] = None
        if sop.match_type == "score" and sop.score_config:
            score = compute_score(sop.score_config, augmented_facts)

        result = MatchResult(
            sop_id=sop.id,
            sop_title=sop.title,
            category=sop.category,
            severity=sop.severity,
            overrides=sop.overrides,
            priority=sop.priority,
            matched_conditions=matched,
            advice_template=sop.advice_template,
            score=score,
        )
        results.append(result)
        logger.info("Matched %s (%s)", sop.id, sop.severity)

    return results
