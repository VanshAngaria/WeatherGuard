"""
Tests for the policy selector (conflict resolution).
All tests are deterministic — no LLM, no API calls.
"""

import pytest

from app.policy.models import MatchResult
from app.policy.selector import resolve_policy


def _make_match(sop_id: str, severity: str, overrides: bool = False, priority: int = 50) -> MatchResult:
    return MatchResult(
        sop_id=sop_id,
        sop_title=f"Title {sop_id}",
        category="outdoor_exercise",
        severity=severity,
        overrides=overrides,
        priority=priority,
        matched_conditions={"test_field": 42},
        advice_template="Test advice.",
    )


# ---------------------------------------------------------------------------
# Basic resolution
# ---------------------------------------------------------------------------

def test_single_match():
    """One match → it becomes primary with empty secondary."""
    m = _make_match("SOP-001", "critical")
    decision = resolve_policy([m])
    assert decision.primary.sop_id == "SOP-001"
    assert decision.secondary_matches == []


def test_severity_wins():
    """Higher severity wins when no overrides."""
    matches = [
        _make_match("SOP-002", "high", priority=20),
        _make_match("SOP-001", "critical", priority=10),
        _make_match("SOP-008", "low", priority=50),
    ]
    decision = resolve_policy(matches)
    assert decision.primary.sop_id == "SOP-001"
    assert len(decision.secondary_matches) == 2


def test_priority_tiebreaker():
    """On severity tie, lower priority number wins."""
    matches = [
        _make_match("SOP-A", "high", priority=30),
        _make_match("SOP-B", "high", priority=10),
        _make_match("SOP-C", "high", priority=20),
    ]
    decision = resolve_policy(matches)
    assert decision.primary.sop_id == "SOP-B"


def test_override_wins_over_higher_severity():
    """Override SOP wins even if a non-override has same severity."""
    matches = [
        _make_match("SOP-001", "critical", overrides=False, priority=10),
        _make_match("SOP-005", "critical", overrides=True, priority=1),
    ]
    decision = resolve_policy(matches)
    assert decision.primary.sop_id == "SOP-005"


def test_override_wins_when_non_override_is_same_or_higher_severity():
    """Override SOP at 'high' wins over non-override at 'critical'."""
    matches = [
        _make_match("SOP-001", "critical", overrides=False, priority=10),
        _make_match("SOP-019", "high", overrides=True, priority=2),
    ]
    decision = resolve_policy(matches)
    # Only override SOPs are eligible when an override exists
    assert decision.primary.sop_id == "SOP-019"
    # Non-override becomes secondary
    assert any(m.sop_id == "SOP-001" for m in decision.secondary_matches)


def test_multiple_overrides_resolved_by_priority():
    """Two override SOPs → lower priority number wins."""
    matches = [
        _make_match("SOP-005", "critical", overrides=True, priority=1),
        _make_match("SOP-019", "critical", overrides=True, priority=2),
    ]
    decision = resolve_policy(matches)
    assert decision.primary.sop_id == "SOP-005"


def test_empty_matches_raises():
    """resolve_policy raises ValueError on empty list."""
    with pytest.raises(ValueError):
        resolve_policy([])


def test_resolution_reason_populated():
    """resolution_reason string is non-empty."""
    matches = [_make_match("SOP-001", "critical")]
    decision = resolve_policy(matches)
    assert len(decision.resolution_reason) > 0
