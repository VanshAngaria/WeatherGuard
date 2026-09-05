"""
SOP YAML loader.
Reads policies/sops.yaml and returns validated List[SOP] instances.
No eval/exec is used anywhere.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml

from app.policy.models import SOP

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).parent.parent.parent / "policies" / "sops.yaml"


def load_sops(policy_path: Path = _DEFAULT_POLICY_PATH) -> List[SOP]:
    """
    Load and validate all SOPs from the YAML file.

    Args:
        policy_path: Absolute path to the sops.yaml file.

    Returns:
        List of validated SOP instances.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        ValueError: If the YAML is malformed or a SOP fails validation.
    """
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
            logger.debug("Loaded SOP: %s — %s", sop.id, sop.title)
        except Exception as exc:
            raise ValueError(f"Invalid SOP entry {entry.get('id', '?')}: {exc}") from exc

    logger.info("Loaded %d SOPs from %s", len(sops), policy_path)
    return sops


@lru_cache(maxsize=1)
def get_sops() -> List[SOP]:
    """
    Return the cached SOP list loaded from the default policy file.
    Cache is invalidated on interpreter restart (in-process only).
    """
    return load_sops()


def get_all_required_fields() -> List[str]:
    """
    Return the union of all required_fields across every loaded SOP.
    Used by the weather fetcher to dynamically build its API request.
    Adding a new SOP with new required_fields automatically propagates here.
    """
    sops = get_sops()
    fields: set[str] = set()
    for sop in sops:
        fields.update(sop.required_fields)
    return sorted(fields)
