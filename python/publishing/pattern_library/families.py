from __future__ import annotations

import re
from typing import List


def normalize_pattern_slug(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "pattern"


def _rows(mask81: str) -> List[str]:
    return [mask81[i : i + 9] for i in range(0, 81, 9)]


def _cols(mask81: str) -> List[str]:
    rows = _rows(mask81)
    return ["".join(rows[r][c] for r in range(9)) for c in range(9)]


def _count_ones(mask81: str) -> int:
    return sum(1 for ch in mask81 if ch == "1")


def _is_vertically_symmetric(mask81: str) -> bool:
    rows = _rows(mask81)
    return all(row == row[::-1] for row in rows)


def _is_horizontally_symmetric(mask81: str) -> bool:
    rows = _rows(mask81)
    return rows == rows[::-1]


def _is_rotationally_symmetric(mask81: str) -> bool:
    rows = _rows(mask81)
    rotated = [row[::-1] for row in rows[::-1]]
    return rows == rotated


def _is_main_diagonal_symmetric(mask81: str) -> bool:
    rows = _rows(mask81)
    for r in range(9):
        for c in range(9):
            if rows[r][c] != rows[c][r]:
                return False
    return True


def infer_symmetry_type(mask81: str) -> str:
    vertical = _is_vertically_symmetric(mask81)
    horizontal = _is_horizontally_symmetric(mask81)
    rotational = _is_rotationally_symmetric(mask81)
    diagonal = _is_main_diagonal_symmetric(mask81)

    if vertical and horizontal and rotational:
        return "full"
    if rotational:
        return "rotational"
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    if diagonal:
        return "diagonal"
    return "none"


def infer_visual_family(mask81: str, explicit_tags: List[str] | None = None) -> str:
    tags = {str(t).strip().lower() for t in (explicit_tags or []) if str(t).strip()}
    if "heart" in tags:
        return "heart"
    if "diamond" in tags:
        return "geometric"
    if "letter" in tags:
        return "letterform"
    if "seasonal" in tags:
        return "seasonal"

    clue_count = _count_ones(mask81)
    symmetry = infer_symmetry_type(mask81)

    if symmetry in {"vertical", "horizontal", "rotational", "full", "diagonal"}:
        if 28 <= clue_count <= 40:
            return "classic_symmetric"
        return "geometric"

    if clue_count <= 26:
        return "sparse"
    if clue_count >= 55:
        return "dense"

    return "decorative"