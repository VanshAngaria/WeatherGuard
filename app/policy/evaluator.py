"""
Deterministic SOP condition evaluator.
Evaluates the condition AST from a SOP against a WeatherFacts dict.

DESIGN RULES:
- Never uses eval() or exec().
- A missing fact (None) always causes a leaf test to return False.
- Score-type conditions use an explicit weighted scoring function.
- No LLM call at any point in this module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.policy.models import (
    AllCondition,
    AnyCondition,
    ConditionNode,
    LeafCondition,
    NotCondition,
    ScoreConfig,
    ScoreGteCondition,
    ScoreLtCondition,
    SOP,
)

logger = logging.getLogger(__name__)

# Type alias: a flat dict of weather field values, possibly None.
FactsDict = Dict[str, Optional[Any]]


# ---------------------------------------------------------------------------
# Leaf operators
# ---------------------------------------------------------------------------

def _eval_leaf(condition: LeafCondition, facts: FactsDict) -> bool:
    """
    Evaluate a single leaf condition against the facts dict.
    Returns False if the required field is missing (None).
    """
    raw = facts.get(condition.field)
    if raw is None:
        logger.debug(
            "Leaf %s %s %s → False (field missing/None)",
            condition.field, condition.op, condition.value,
        )
        return False

    val = raw
    threshold = condition.value
    op = condition.op

    try:
        if op == "gt":
            result = float(val) > float(threshold)
        elif op == "gte":
            result = float(val) >= float(threshold)
        elif op == "lt":
            result = float(val) < float(threshold)
        elif op == "lte":
            result = float(val) <= float(threshold)
        elif op == "eq":
            result = val == threshold
        elif op == "ne":
            result = val != threshold
        elif op == "in":
            result = val in threshold
        elif op == "contains":
            result = threshold in val
        elif op == "between":
            lo, hi = float(threshold[0]), float(threshold[1])
            result = lo <= float(val) <= hi
        else:
            logger.warning("Unknown operator '%s' in leaf condition.", op)
            result = False
    except (TypeError, ValueError) as exc:
        logger.warning("Leaf evaluation error for field '%s': %s", condition.field, exc)
        result = False

    logger.debug(
        "Leaf %s(%s) %s %s → %s",
        condition.field, val, op, threshold, result,
    )
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(score_config: ScoreConfig, facts: FactsDict) -> float:
    """
    Compute the weighted suitability score for a score-type SOP.
    Each criterion is evaluated as a leaf; if it passes, its weight is added.
    Missing fields score 0 for that criterion (consistent with leaf behaviour).
    """
    total = 0.0
    for criterion in score_config.criteria:
        leaf = LeafCondition(
            type="leaf",
            field=criterion.field,
            op=criterion.op,
            value=criterion.value,
        )
        if _eval_leaf(leaf, facts):
            total += criterion.weight
    logger.debug("Score computed: %.1f", total)
    return total


# ---------------------------------------------------------------------------
# Recursive condition evaluator
# ---------------------------------------------------------------------------

def evaluate_condition(condition: ConditionNode, facts: FactsDict, score_config: Optional[ScoreConfig] = None) -> bool:
    """
    Recursively evaluate a condition node against the facts dict.

    Args:
        condition: Any condition node from the SOP AST.
        facts: Flat dict of weather fact values.
        score_config: Required for score_gte / score_lt nodes.

    Returns:
        True if the condition is satisfied, False otherwise.
    """
    cond_type = condition.type

    if cond_type == "leaf":
        return _eval_leaf(condition, facts)

    elif cond_type == "all":
        return all(
            evaluate_condition(sub, facts, score_config)
            for sub in condition.conditions
        )

    elif cond_type == "any":
        return any(
            evaluate_condition(sub, facts, score_config)
            for sub in condition.conditions
        )

    elif cond_type == "not":
        return not evaluate_condition(condition.condition, facts, score_config)

    elif cond_type == "score_gte":
        if score_config is None:
            logger.error("score_gte condition requires score_config but none provided.")
            return False
        score = compute_score(score_config, facts)
        return score >= condition.threshold

    elif cond_type == "score_lt":
        if score_config is None:
            logger.error("score_lt condition requires score_config but none provided.")
            return False
        score = compute_score(score_config, facts)
        return score < condition.threshold

    else:
        logger.warning("Unknown condition type: %s", cond_type)
        return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def sop_matches(sop: SOP, facts: FactsDict) -> bool:
    """
    Return True if the SOP's condition is satisfied by the given facts.
    Handles both rule-type and score-type SOPs uniformly.
    """
    return evaluate_condition(sop.condition, facts, sop.score_config)


def get_matched_fields(sop: SOP, facts: FactsDict) -> Dict[str, Any]:
    """
    Return a dict of the relevant field values that caused the SOP to match.
    Used to populate matched_conditions in MatchResult.
    """
    return {
        field: facts.get(field)
        for field in sop.required_fields
        if facts.get(field) is not None
    }
