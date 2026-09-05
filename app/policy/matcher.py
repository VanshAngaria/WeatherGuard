"""
SOP matcher.
Iterates over all loaded SOPs, checks category applicability,
evaluates the condition, and returns every matching SOP as a MatchResult.

No LLM call here. Purely deterministic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from app.policy.evaluator import get_matched_fields, sop_matches, compute_score
from app.policy.loader import get_sops
from app.policy.models import MatchResult, SOP

logger = logging.getLogger(__name__)

# Map intent categories → SOP applies_to_categories
CATEGORY_ALIASES: Dict[str, List[str]] = {
    "outdoor_exercise": ["outdoor_exercise"],
    "outdoor_recreation": ["outdoor_recreation"],
    "travel": ["travel"],
    "vulnerable_groups": ["vulnerable_groups"],
    "general": ["general"],
    "water_activities": ["water_activities"],
}


def _category_applies(sop: SOP, intent_categories: List[str]) -> bool:
    """
    Return True if this SOP is applicable for at least one of the intent categories.
    A SOP with applies_to_categories=["*"] applies to everything.
    """
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
    Run every SOP against the provided facts and intent context.

    Args:
        facts:              Flat dict of WeatherFacts values.
        intent_categories:  List of semantic categories from intent parser.
        mode:               Travel/activity mode (e.g. "cycling", "scooter").
        group:              Vulnerable group flag (e.g. "children").

    Returns:
        List of MatchResult for every SOP whose condition is satisfied.
        Empty list if nothing matches.
    """
    sops = _sops_override if _sops_override is not None else get_sops()
    results: List[MatchResult] = []

    # Inject mode/group into facts so conditions can test them
    augmented_facts = dict(facts)
    if mode:
        augmented_facts["mode"] = mode
    if group:
        augmented_facts["group"] = group

    for sop in sops:
        # --- Category filter ---
        if not _category_applies(sop, intent_categories):
            logger.debug("SOP %s skipped (category mismatch: %s vs %s)", sop.id, sop.applies_to_categories, intent_categories)
            continue

        # --- Condition evaluation ---
        if not sop_matches(sop, augmented_facts):
            logger.debug("SOP %s condition not satisfied.", sop.id)
            continue

        # --- Collect matched field values ---
        matched = get_matched_fields(sop, augmented_facts)

        # --- Score (for score-type SOPs) ---
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
        logger.info(
            "SOP %s MATCHED | severity=%s | override=%s | conditions=%s",
            sop.id, sop.severity, sop.overrides, matched,
        )

    logger.info(
        "match_sops complete: %d/%d SOPs matched for categories=%s",
        len(results), len(sops), intent_categories,
    )
    return results
