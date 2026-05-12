from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from python.publishing.difficulty.technique_difficulty_map import (
    DIFFICULTY_VERSION,
    EASY,
    EXPERT,
    GENIUS,
    HARD,
    MEDIUM,
    VERY_HARD,
)

_ALLOWED_DIFFICULTIES = {EASY, MEDIUM, HARD, VERY_HARD, EXPERT, GENIUS}


def normalize_techniques_difficulty(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        v = str(value).strip().lower()
        if not v:
            continue
        if v not in _ALLOWED_DIFFICULTIES:
            raise ValueError(f"Unsupported technique difficulty: {value}")
        out.append(v)
    return out


def derive_puzzle_difficulty(
    techniques_difficulty: Sequence[str],
) -> str:
    """
    Derive the user-facing puzzle difficulty.

    Rules:
    - no techniques or only easy techniques -> easy
    - at least one medium, and no hard/very_hard/expert -> medium
    - at least one hard, but no very_hard/expert -> hard
    - expert if:
        * exactly one expert and at most one very_hard, or
        * multiple very_hard techniques, or
        * at least one very_hard plus multiple hard techniques
    - genius if:
        * genius appears directly, or
        * multiple expert techniques, or
        * at least one expert plus multiple very_hard techniques
    """
    normalized = normalize_techniques_difficulty(techniques_difficulty)

    if not normalized:
        return EASY

    counts = Counter(normalized)
    hard_count = counts[HARD]
    very_hard_count = counts[VERY_HARD]
    expert_count = counts[EXPERT]
    genius_count = counts[GENIUS]

    # Defensive fallback: if GENIUS already appears directly in the
    # technique-difficulty stream, preserve the top tier.
    if genius_count > 0:
        return GENIUS

    # Multiple expert-level pressure points remain genius.
    if expert_count >= 2:
        return GENIUS

    # One expert plus multiple very-hard techniques remains genius.
    if expert_count >= 1 and very_hard_count >= 2:
        return GENIUS

    # One expert with no more than one very-hard technique is expert.
    if expert_count >= 1 and very_hard_count <= 1:
        return EXPERT

    # Multiple very-hard pressure points are expert.
    if very_hard_count >= 2:
        return EXPERT

    # One very-hard technique plus multiple hard techniques is expert.
    if very_hard_count >= 1 and hard_count >= 2:
        return EXPERT

    # Otherwise, hard/very-hard techniques without enough pressure remain hard.
    if hard_count > 0 or very_hard_count > 0:
        return HARD

    if counts[MEDIUM] > 0:
        return MEDIUM

    return EASY


def build_puzzle_difficulty_payload(
    techniques_difficulty: Sequence[str],
) -> dict[str, object]:
    normalized = normalize_techniques_difficulty(techniques_difficulty)
    return {
        "techniques_difficulty": normalized,
        "puzzle_difficulty": derive_puzzle_difficulty(normalized),
        "difficulty_version": DIFFICULTY_VERSION,
    }