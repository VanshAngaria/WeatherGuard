from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class LeafCondition(BaseModel):
    """A single boolean comparison on a specific weather field."""
    type: Literal["leaf"]
    field: str
    op: Literal["gt", "gte", "lt", "lte", "eq", "ne", "in", "contains", "between"]
    value: Any


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
    """Fires when the cumulative score meets or exceeds a threshold."""
    type: Literal["score_gte"]
    threshold: float


class ScoreLtCondition(BaseModel):
    """Fires when the cumulative score is strictly below a threshold."""
    type: Literal["score_lt"]
    threshold: float


ConditionNode = Union[
    LeafCondition,
    AllCondition,
    AnyCondition,
    NotCondition,
    ScoreGteCondition,
    ScoreLtCondition,
]

AllCondition.model_rebuild()
AnyCondition.model_rebuild()
NotCondition.model_rebuild()


class ScoreCriterion(BaseModel):
    """Scoring rule with a weight."""
    field: str
    op: Literal["gt", "gte", "lt", "lte", "eq", "ne", "in", "contains", "between"]
    value: Any
    weight: float


class ScoreConfig(BaseModel):
    """Configuration for score-based SOP evaluation."""
    criteria: List[ScoreCriterion]


class SOP(BaseModel):
    """Safety Operating Procedure definition loaded from sops.yaml."""
    id: str
    title: str
    category: str
    severity: Literal["low", "moderate", "high", "critical"]
    match_type: Literal["rule", "score"]
    overrides: bool = False
    priority: int = 99
    applies_to_categories: List[str] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    score_config: Optional[ScoreConfig] = None
    condition: ConditionNode
    advice_template: str


class MatchResult(BaseModel):
    """Result of an individual matching SOP."""
    sop_id: str
    sop_title: str
    category: str
    severity: Literal["low", "moderate", "high", "critical"]
    overrides: bool
    priority: int
    matched_conditions: Dict[str, Any]
    condition_traces: List[Dict[str, Any]] = Field(default_factory=list)
    advice_template: str
    score: Optional[float] = None


class PolicyDecision(BaseModel):
    """Outcome of conflict resolution selecting primary and secondary SOPs."""
    primary: MatchResult
    secondary_matches: List[MatchResult] = Field(default_factory=list)
    resolution_reason: str
