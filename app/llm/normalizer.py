"""
Input Normalizer — lightweight typo/grammar correction for weather-advisory queries.

Philosophy:
- Correct HIGH-CONFIDENCE activity spelling mistakes silently.
- Expand clearly informal/incomplete phrasing to parseable form.
- NEVER invent activity, location, time, or group.
- Return (normalized_message, interpreted_as) where interpreted_as is
  non-None only when the message was meaningfully changed.

Uses:
- difflib.get_close_matches for activity word correction (no external deps)
- Simple pattern expansion for common informal structures
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known activity vocabulary (for close-match correction)
# ---------------------------------------------------------------------------

_ACTIVITY_WORDS = [
    "cycle", "cycling", "bicycle", "bike", "biking",
    "walk", "walking",
    "run", "running", "jog", "jogging",
    "hike", "hiking",
    "drive", "driving",
    "scooter", "motorbike", "motorcycle",
    "swim", "swimming",
    "picnic", "outing",
    "park", "playground",
    "commute", "commuting", "travel",
    "skydiving",  # known unsupported activity — still correct the spelling
]

# Canonical mapping: normalized form → standard word
_CANONICAL = {
    "bike": "cycle",
    "biking": "cycling",
    "bicycle": "cycle",
    "jog": "run",
    "jogging": "running",
    "drive": "drive",
    "motorcycle": "motorbike",
    "playground": "park",
}

# Short informal expansions for common patterns
_INFORMAL_PATTERNS = [
    # "can i walk roorkee now" → "Can I walk in Roorkee now?"
    (
        re.compile(r"^can i (\w+)\s+([a-zA-Z\s]+?)(\s+now|\s+today|\s+tonight)?$", re.I),
        lambda m: f"Can I {m.group(1)} in {m.group(2).strip()}{m.group(3) or ''}?",
    ),
    # "is cycling safe delhi today" → "Is cycling safe in Delhi today?"
    (
        re.compile(r"^is (\w+(?:ing)?)\s+safe\s+([a-zA-Z\s]+?)(\s+today|\s+now|\s+tonight)?$", re.I),
        lambda m: f"Is {m.group(1)} safe in {m.group(2).strip()}{m.group(3) or ''}?",
    ),
    # "should i take my kid park delhi today"
    (
        re.compile(r"^should i take my kid(?:s)?\s+(\w+)\s+([a-zA-Z\s]+?)(\s+today|\s+now)?$", re.I),
        lambda m: f"Should I take my child to the {m.group(1)} in {m.group(2).strip()}{m.group(3) or ''}?",
    ),
]


def _correct_activity_typos(message: str) -> Tuple[str, bool]:
    """
    Correct obvious activity spelling typos using difflib close-match.
    Returns (corrected_message, was_changed).
    Only corrects if there is exactly one close match with cutoff >= 0.82.
    """
    words = message.split()
    corrected = []
    changed = False

    for word in words:
        clean = word.strip("?!.,;:\"'").lower()
        if len(clean) < 4:
            corrected.append(word)
            continue

        matches = difflib.get_close_matches(clean, _ACTIVITY_WORDS, n=1, cutoff=0.82)
        if matches and matches[0] != clean:
            # Use canonical form if available
            canonical = _CANONICAL.get(matches[0], matches[0])
            # Preserve original casing of first letter
            replacement = canonical if not word[0].isupper() else canonical.capitalize()
            corrected.append(replacement)
            changed = True
            logger.debug("Typo correction: '%s' → '%s'", word, replacement)
        else:
            corrected.append(word)

    return " ".join(corrected), changed


def _expand_informal(message: str) -> Tuple[str, bool]:
    """
    Expand clearly informal / incomplete phrases to parseable English.
    Returns (expanded, was_changed).
    """
    stripped = message.strip()
    for pattern, expander in _INFORMAL_PATTERNS:
        m = pattern.match(stripped)
        if m:
            try:
                expanded = expander(m)
                if expanded.lower() != stripped.lower():
                    logger.debug("Informal expansion: '%s' → '%s'", stripped, expanded)
                    return expanded, True
            except Exception:
                pass
    return message, False


def normalize_input(message: str) -> Tuple[str, Optional[str]]:
    """
    Main entry point. Normalize a user message.

    Returns:
        (normalized_message, interpreted_as)
        - normalized_message: the cleaned/corrected message to send to the LLM
        - interpreted_as: human-readable "Interpreted as: …" string if changed, else None

    Guarantees:
        - Will NOT invent activity, location, time, or group.
        - Changes are conservative (high-confidence corrections only).
        - If uncertain, returns original unchanged.
    """
    if not message or not message.strip():
        return message, None

    original = message.strip()
    current = original
    any_changed = False

    # Step 1: Fix obvious activity spelling typos
    current, changed1 = _correct_activity_typos(current)
    if changed1:
        any_changed = True

    # Step 2: Expand informal structure only on short messages (< 10 words)
    # to avoid mangling well-formed sentences
    if len(current.split()) <= 10:
        current, changed2 = _expand_informal(current)
        if changed2:
            any_changed = True

    if any_changed and current.lower() != original.lower():
        interpreted_as = current
        logger.info("Input normalized: '%s' → '%s'", original, current)
        return current, interpreted_as

    return original, None
