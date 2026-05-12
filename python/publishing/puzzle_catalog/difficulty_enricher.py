from __future__ import annotations

from typing import Iterable

from python.publishing.difficulty.puzzle_difficulty import build_puzzle_difficulty_payload
from python.publishing.difficulty.technique_difficulty_map import get_technique_difficulty


def build_techniques_difficulty(techniques_used: Iterable[str]) -> list[str]:
    values: list[str] = []
    for technique_name in techniques_used:
        difficulty = get_technique_difficulty(technique_name)
        if difficulty is not None:
            values.append(difficulty)
    return values


def enrich_candidate_difficulty(
    *,
    techniques_used: Iterable[str],
) -> dict[str, object]:
    techniques_difficulty = build_techniques_difficulty(techniques_used)
    payload = build_puzzle_difficulty_payload(techniques_difficulty)
    return payload