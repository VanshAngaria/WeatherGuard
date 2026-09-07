from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.policy.models import (
    ConditionNode,
    LeafCondition,
    ScoreConfig,
    SOP,
)

logger = logging.getLogger(__name__)

FactsDict = Dict[str, Optional[Any]]

OP_SYMBOLS = {
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "eq": "==",
    "ne": "!=",
    "in": "in",
    "contains": "contains",
    "between": "between",
}


def _eval_leaf(condition: LeafCondition, facts: FactsDict) -> bool:
    """Evaluate a single leaf condition against the facts dictionary."""
    raw = facts.get(condition.field)
    if raw is None:
        return False

    val = raw
    threshold = condition.value
    op = condition.op

    try:
        if op == "gt":
            return float(val) > float(threshold)
        elif op == "gte":
            return float(val) >= float(threshold)
        elif op == "lt":
            return float(val) < float(threshold)
        elif op == "lte":
            return float(val) <= float(threshold)
        elif op == "eq":
            return val == threshold
        elif op == "ne":
            return val != threshold
        elif op == "in":
            return val in threshold
        elif op == "contains":
            return threshold in val
        elif op == "between":
            lo, hi = float(threshold[0]), float(threshold[1])
            return lo <= float(val) <= hi
        else:
            logger.warning("Unknown operator '%s' in leaf condition.", op)
            return False
    except (TypeError, ValueError) as exc:
        logger.warning("Leaf evaluation error on '%s': %s", condition.field, exc)
        return False


def compute_score(score_config: ScoreConfig, facts: FactsDict) -> float:
    """Compute the weighted suitability score for a score-type SOP."""
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
    return total


def evaluate_condition(condition: ConditionNode, facts: FactsDict, score_config: Optional[ScoreConfig] = None) -> bool:
    """Recursively evaluate a condition node against the facts dictionary."""
    cond_type = condition.type

    if cond_type == "leaf":
        return _eval_leaf(condition, facts)
    elif cond_type == "all":
        return all(evaluate_condition(sub, facts, score_config) for sub in condition.conditions)
    elif cond_type == "any":
        return any(evaluate_condition(sub, facts, score_config) for sub in condition.conditions)
    elif cond_type == "not":
        return not evaluate_condition(condition.condition, facts, score_config)
    elif cond_type == "score_gte":
        if score_config is None:
            return False
        return compute_score(score_config, facts) >= condition.threshold
    elif cond_type == "score_lt":
        if score_config is None:
            return False
        return compute_score(score_config, facts) < condition.threshold
    else:
        logger.warning("Unknown condition type: %s", cond_type)
        return False


def sop_matches(sop: SOP, facts: FactsDict) -> bool:
    """Return True if the SOP condition is satisfied by facts."""
    return evaluate_condition(sop.condition, facts, sop.score_config)


def _collect_leaf_conditions(condition: ConditionNode) -> List[LeafCondition]:
    """Recursively gather all leaf condition nodes."""
    leaves: List[LeafCondition] = []
    if condition.type == "leaf":
        leaves.append(condition)
    elif condition.type in ("all", "any"):
        for sub in condition.conditions:
            leaves.extend(_collect_leaf_conditions(sub))
    elif condition.type == "not":
        leaves.extend(_collect_leaf_conditions(condition.condition))
    return leaves


def get_matched_condition_trace(sop: SOP, facts: FactsDict) -> List[Dict[str, Any]]:
    """
    Extract structured trace items explaining exactly why this SOP matched:
    field name, observed value, operator, and threshold value.
    """
    trace = []
    leaves = _collect_leaf_conditions(sop.condition)
    for leaf in leaves:
        val = facts.get(leaf.field)
        if val is not None:
            trace.append({
                "field": leaf.field,
                "observed": val,
                "operator": OP_SYMBOLS.get(leaf.op, leaf.op),
                "threshold": leaf.value,
                "satisfied": _eval_leaf(leaf, facts),
            })
    return trace


def get_matched_fields(sop: SOP, facts: FactsDict) -> Dict[str, Any]:
    """Extract the specific fact values relevant to this SOP."""
    return {
        field: facts.get(field)
        for field in sop.required_fields
        if facts.get(field) is not None
    }
