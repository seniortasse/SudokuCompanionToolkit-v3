from __future__ import annotations

from typing import List

from python.publishing.ids.validators import is_valid_library_id, is_valid_pattern_id
from python.publishing.schemas.models import PatternRecord


def validate_pattern_record(pattern: PatternRecord) -> List[str]:
    errors: List[str] = []

    if not is_valid_pattern_id(pattern.pattern_id):
        errors.append(f"Invalid pattern_id: {pattern.pattern_id}")

    if not is_valid_library_id(pattern.library_id):
        errors.append(f"Invalid library_id: {pattern.library_id}")

    if pattern.grid_size <= 0:
        errors.append("grid_size must be > 0")

    if pattern.layout_type.strip() == "":
        errors.append("layout_type must not be blank")

    if len(pattern.mask81) != 81:
        errors.append("mask81 must be exactly 81 characters long for classic 9x9 assets")

    bad_mask_chars = [ch for ch in pattern.mask81 if ch not in {"0", "1"}]
    if bad_mask_chars:
        errors.append("mask81 must contain only '0' and '1' characters")

    computed_clues = sum(1 for ch in pattern.mask81 if ch == "1")
    if pattern.clue_count != computed_clues:
        errors.append(
            f"clue_count mismatch: declared={pattern.clue_count}, computed={computed_clues}"
        )

    if pattern.name.strip() == "":
        errors.append("name must not be blank")

    if pattern.slug.strip() == "":
        errors.append("slug must not be blank")

    return errors