"""
Policy data models.
Defines Pydantic models for SOPs, match results, and related structures.
The LLM never touches these — they are used exclusively by the deterministic engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Condition AST
# ---------------------------------------------------------------------------

class LeafCondition(BaseModel):
    """A single boolean test on one weather field."""
    type: Literal["leaf"]
    field: str
    op: Literal["gt", "gte", "lt", "lte", "eq", "ne", "in", "contains", "between"]
    value: Any  # number, string, list


class AllCondition(BaseModel):
    """Logical AND of child conditions."""
    type: Literal["all"]
    conditions: List["ConditionNode"]


class AnyCondition(BaseModel):
    """Logical OR of child conditions."""
    type: Literal["any"]
    conditions: List["ConditionNode"]


class NotCondition(BaseModel):
    """Logical NOT of a child condition."""
    type: Literal["not"]
    condition: "ConditionNode"


class ScoreGteCondition(BaseModel):
    """
    Score-based condition: fires when the weighted score >= threshold.
    Actual scoring is driven by the SOP's score_config.
    """
    type: Literal["score_gte"]
    threshold: float


class ScoreLtCondition(BaseModel):
    """
    Score-based condition: fires when the weighted score < threshold.
    Actual scoring is driven by the SOP's score_config.
    """
    type: Literal["score_lt"]
    threshold: float


# Union of all possible condition types (distinct name to avoid shadowing)
ConditionNode = Union[
    LeafCondition,
    AllCondition,
    AnyCondition,
    NotCondition,
    ScoreGteCondition,
    ScoreLtCondition,
]

# Rebuild models to resolve forward references
AllCondition.model_rebuild()
AnyCondition.model_rebuild()
NotCondition.model_rebuild()


# ---------------------------------------------------------------------------
# Score config (used by SOP-008 / SOP-008B)
# ---------------------------------------------------------------------------

class ScoreCriterion(BaseModel):
    """One scoring rule within a score_config block."""
    field: str
    op: Literal["gt", "gte", "lt", "lte", "eq", "ne", "in", "contains", "between"]
    value: Any
    weight: float


class ScoreConfig(BaseModel):
    """Full scoring configuration attached to a score-type SOP."""
    criteria: List[ScoreCriterion]


# ---------------------------------------------------------------------------
# SOP
# ---------------------------------------------------------------------------

class SOP(BaseModel):
    """A single Safety Operating Procedure loaded from sops.yaml."""
    id: str
    title: str
    category: str
    severity: Literal["low", "moderate", "high", "critical"]
    match_type: Literal["rule", "score"]
    overrides: bool = False
    priority: int = 99
    applies_to_categories: List[str] = Field(default_factory=lambda: [])
    required_fields: List[str] = Field(default_factory=list)
    score_config: Optional[ScoreConfig] = None
    condition: ConditionNode
    advice_template: str


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

class MatchResult(BaseModel):
    """Produced by the policy matcher for every SOP that fires."""
    sop_id: str
    sop_title: str
    category: str
    severity: Literal["low", "moderate", "high", "critical"]
    overrides: bool
    priority: int
    matched_conditions: Dict[str, Any]  # field_name → actual value
    advice_template: str
    score: Optional[float] = None  # populated for score-type SOPs


# ---------------------------------------------------------------------------
# Policy resolution result
# ---------------------------------------------------------------------------

class PolicyDecision(BaseModel):
    """Output of the conflict-resolution step."""
    primary: MatchResult
    secondary_matches: List[MatchResult] = Field(default_factory=list)
    resolution_reason: str
