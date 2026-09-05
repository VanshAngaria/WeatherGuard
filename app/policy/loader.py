from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml

from app.policy.models import SOP

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policies" / "sops.yaml"


def load_sops(policy_path: Path = _DEFAULT_POLICY_PATH) -> List[SOP]:
    """Load and validate all SOP definitions from YAML."""
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    with open(policy_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "sops" not in raw:
        raise ValueError("Policy YAML must have a top-level 'sops' key.")

    sops: List[SOP] = []
    for entry in raw["sops"]:
        try:
            sop = SOP.model_validate(entry)
            sops.append(sop)
        except Exception as exc:
            raise ValueError(f"Invalid SOP entry {entry.get('id', '?')}: {exc}") from exc

    logger.info("Loaded %d SOPs from %s", len(sops), policy_path)
    return sops


@lru_cache(maxsize=1)
def get_sops() -> List[SOP]:
    """Return cached SOP list."""
    return load_sops()


def get_all_required_fields() -> List[str]:
    """Return union of all required_fields across all active SOPs."""
    sops = get_sops()
    fields: set[str] = set()
    for sop in sops:
        fields.update(sop.required_fields)
    return sorted(fields)
