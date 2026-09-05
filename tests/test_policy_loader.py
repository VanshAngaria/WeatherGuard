"""
Tests for the SOP policy loader.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from app.policy.loader import load_sops


def test_load_default_sops():
    """All 12 SOPs load without error."""
    sops = load_sops()
    assert len(sops) == 12
    ids = {s.id for s in sops}
    expected = {
        "SOP-001", "SOP-002", "SOP-004", "SOP-005",
        "SOP-007", "SOP-008", "SOP-008B", "SOP-010",
        "SOP-012", "SOP-014", "SOP-015", "SOP-019",
    }
    assert ids == expected


def test_sop_required_fields():
    """Every SOP has at least one required_field."""
    sops = load_sops()
    for sop in sops:
        assert len(sop.required_fields) >= 1, f"{sop.id} has no required_fields"


def test_sop_severity_values():
    """All severity values are in the valid set."""
    valid = {"low", "moderate", "high", "critical"}
    sops = load_sops()
    for sop in sops:
        assert sop.severity in valid, f"{sop.id} has invalid severity: {sop.severity}"


def test_override_sops_have_priority():
    """Override SOPs have priority <= 2."""
    sops = load_sops()
    overrides = [s for s in sops if s.overrides]
    assert len(overrides) >= 2, "Expected at least 2 override SOPs (SOP-005, SOP-019)"
    for s in overrides:
        assert s.priority <= 2, f"{s.id} is override but priority={s.priority}"


def test_score_type_sops_have_config():
    """Score-type SOPs have score_config populated."""
    sops = load_sops()
    score_sops = [s for s in sops if s.match_type == "score"]
    assert len(score_sops) >= 2, "Expected SOP-008 and SOP-008B"
    for s in score_sops:
        assert s.score_config is not None, f"{s.id} is score-type but has no score_config"


def test_missing_file_raises():
    """FileNotFoundError on missing policy file."""
    with pytest.raises(FileNotFoundError):
        load_sops(Path("/nonexistent/sops.yaml"))


def test_malformed_yaml_raises():
    """ValueError on invalid SOP schema."""
    bad_yaml = {"sops": [{"id": "BAD", "title": "No condition"}]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(bad_yaml, tmp)
        tmp_path = Path(tmp.name)
    with pytest.raises(ValueError):
        load_sops(tmp_path)
    tmp_path.unlink()


def test_dynamic_required_fields():
    """get_all_required_fields returns union across all SOPs."""
    from app.policy.loader import get_all_required_fields
    fields = get_all_required_fields()
    assert "apparent_temperature" in fields
    assert "weathercode" in fields
    assert "visibility" in fields
    assert "uv_index" in fields
    assert "precipitation_sum_today" in fields
