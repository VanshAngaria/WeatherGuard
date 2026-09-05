"""
Tests for the deterministic policy evaluator.
All tests use mocked WeatherFacts — no LLM, no API calls.
"""

import pytest

from app.policy.evaluator import (
    compute_score,
    evaluate_condition,
    sop_matches,
)
from app.policy.loader import load_sops
from app.policy.models import (
    AllCondition,
    AnyCondition,
    LeafCondition,
    NotCondition,
    ScoreConfig,
    ScoreCriterion,
    ScoreGteCondition,
    ScoreLtCondition,
)
from app.weather.models import WeatherFacts


# ---------------------------------------------------------------------------
# Leaf condition tests
# ---------------------------------------------------------------------------

def test_leaf_gte_passes():
    cond = LeafCondition(type="leaf", field="apparent_temperature", op="gte", value=41)
    assert evaluate_condition(cond, {"apparent_temperature": 41.0}) is True
    assert evaluate_condition(cond, {"apparent_temperature": 45.0}) is True


def test_leaf_gte_fails():
    cond = LeafCondition(type="leaf", field="apparent_temperature", op="gte", value=41)
    assert evaluate_condition(cond, {"apparent_temperature": 40.9}) is False


def test_leaf_missing_field_fails():
    """Missing field must NOT cause condition to pass."""
    cond = LeafCondition(type="leaf", field="apparent_temperature", op="gte", value=41)
    assert evaluate_condition(cond, {}) is False
    assert evaluate_condition(cond, {"apparent_temperature": None}) is False


def test_leaf_in_operator():
    cond = LeafCondition(type="leaf", field="weathercode", op="in", value=[95, 96, 99])
    assert evaluate_condition(cond, {"weathercode": 95}) is True
    assert evaluate_condition(cond, {"weathercode": 96}) is True
    assert evaluate_condition(cond, {"weathercode": 1}) is False


def test_leaf_between_operator():
    cond = LeafCondition(type="leaf", field="temperature_2m", op="between", value=[18, 29])
    assert evaluate_condition(cond, {"temperature_2m": 18.0}) is True
    assert evaluate_condition(cond, {"temperature_2m": 25.0}) is True
    assert evaluate_condition(cond, {"temperature_2m": 29.0}) is True
    assert evaluate_condition(cond, {"temperature_2m": 17.9}) is False
    assert evaluate_condition(cond, {"temperature_2m": 30.0}) is False


def test_leaf_lt_operator():
    cond = LeafCondition(type="leaf", field="visibility", op="lt", value=500)
    assert evaluate_condition(cond, {"visibility": 499.0}) is True
    assert evaluate_condition(cond, {"visibility": 500.0}) is False


# ---------------------------------------------------------------------------
# Compound condition tests
# ---------------------------------------------------------------------------

def test_all_condition():
    cond = AllCondition(
        type="all",
        conditions=[
            LeafCondition(type="leaf", field="temperature_2m", op="gte", value=30),
            LeafCondition(type="leaf", field="relative_humidity_2m", op="gte", value=70),
        ],
    )
    assert evaluate_condition(cond, {"temperature_2m": 32.0, "relative_humidity_2m": 75.0}) is True
    assert evaluate_condition(cond, {"temperature_2m": 32.0, "relative_humidity_2m": 60.0}) is False


def test_any_condition():
    cond = AnyCondition(
        type="any",
        conditions=[
            LeafCondition(type="leaf", field="weathercode", op="in", value=[95, 96, 99]),
            LeafCondition(type="leaf", field="wind_gusts_10m", op="gte", value=60),
        ],
    )
    assert evaluate_condition(cond, {"weathercode": 95, "wind_gusts_10m": 30}) is True
    assert evaluate_condition(cond, {"weathercode": 1, "wind_gusts_10m": 65}) is True
    assert evaluate_condition(cond, {"weathercode": 1, "wind_gusts_10m": 30}) is False


def test_not_condition():
    cond = NotCondition(
        type="not",
        condition=LeafCondition(type="leaf", field="weathercode", op="in", value=[95, 96, 99]),
    )
    assert evaluate_condition(cond, {"weathercode": 1}) is True
    assert evaluate_condition(cond, {"weathercode": 95}) is False


# ---------------------------------------------------------------------------
# Score-based tests (SOP-008 / SOP-008B logic)
# ---------------------------------------------------------------------------

def _picnic_score_config() -> ScoreConfig:
    return ScoreConfig(criteria=[
        ScoreCriterion(field="temperature_2m", op="between", value=[18, 29], weight=2),
        ScoreCriterion(field="precipitation_probability", op="lt", value=20, weight=2),
        ScoreCriterion(field="wind_speed_10m", op="lt", value=20, weight=1),
        ScoreCriterion(field="uv_index", op="lt", value=7, weight=1),
    ])


def test_perfect_picnic_score():
    """All 4 criteria pass → score = 6 → SOP-008 fires (>=4), SOP-008B does not (<2)."""
    facts = {
        "temperature_2m": 22.0,
        "precipitation_probability": 5.0,
        "wind_speed_10m": 10.0,
        "uv_index": 4.0,
    }
    cfg = _picnic_score_config()
    score = compute_score(cfg, facts)
    assert score == 6.0

    cond_gte = ScoreGteCondition(type="score_gte", threshold=4)
    cond_lt = ScoreLtCondition(type="score_lt", threshold=2)
    assert evaluate_condition(cond_gte, facts, score_config=cfg) is True
    assert evaluate_condition(cond_lt, facts, score_config=cfg) is False


def test_poor_picnic_score():
    """No criteria pass → score = 0 → SOP-008B fires (<2), SOP-008 does not (>=4)."""
    facts = {
        "temperature_2m": 5.0,
        "precipitation_probability": 90.0,
        "wind_speed_10m": 40.0,
        "uv_index": 9.0,
    }
    cfg = _picnic_score_config()
    score = compute_score(cfg, facts)
    assert score == 0.0

    cond_gte = ScoreGteCondition(type="score_gte", threshold=4)
    cond_lt = ScoreLtCondition(type="score_lt", threshold=2)
    assert evaluate_condition(cond_gte, facts, score_config=cfg) is False
    assert evaluate_condition(cond_lt, facts, score_config=cfg) is True


def test_middle_band_picnic():
    """Middle band (score=2 or 3) — neither SOP-008 nor SOP-008B fires."""
    facts = {
        "temperature_2m": 22.0,       # +2
        "precipitation_probability": 50.0,  # fail
        "wind_speed_10m": 25.0,       # fail
        "uv_index": 9.0,              # fail
    }
    cfg = _picnic_score_config()
    score = compute_score(cfg, facts)
    assert score == 2.0

    cond_gte = ScoreGteCondition(type="score_gte", threshold=4)
    cond_lt = ScoreLtCondition(type="score_lt", threshold=2)
    assert evaluate_condition(cond_gte, facts, score_config=cfg) is False  # SOP-008 does not fire
    assert evaluate_condition(cond_lt, facts, score_config=cfg) is False   # SOP-008B does not fire


# ---------------------------------------------------------------------------
# Full SOP match tests
# ---------------------------------------------------------------------------

def test_sop_001_matches():
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-001")
    assert sop_matches(sop, {"apparent_temperature": 41.0}) is True
    assert sop_matches(sop, {"apparent_temperature": 40.9}) is False


def test_sop_005_thunderstorm_override():
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-005")
    assert sop_matches(sop, {"weathercode": 95}) is True
    assert sop_matches(sop, {"weathercode": 96}) is True
    assert sop_matches(sop, {"weathercode": 99}) is True
    assert sop_matches(sop, {"weathercode": 61}) is False


def test_sop_019_condition_arm1():
    """SOP-019 first arm: heavy rain today + next 2 days."""
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-019")
    facts = {
        "precipitation_sum_today": 70.0,
        "precipitation_sum_next_2d": 90.0,
        "wind_gusts_10m": 20.0,
        "precipitation_probability": 40.0,
    }
    assert sop_matches(sop, facts) is True


def test_sop_019_condition_arm2():
    """SOP-019 second arm: high gusts + high rain probability."""
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-019")
    facts = {
        "precipitation_sum_today": 10.0,
        "precipitation_sum_next_2d": 20.0,
        "wind_gusts_10m": 65.0,
        "precipitation_probability": 70.0,
    }
    assert sop_matches(sop, facts) is True


def test_sop_014_children_heat():
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-014")
    assert sop_matches(sop, {"apparent_temperature": 35.0, "group": "children"}) is True
    assert sop_matches(sop, {"apparent_temperature": 34.9, "group": "children"}) is False
