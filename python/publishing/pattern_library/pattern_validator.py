from __future__ import annotations

from typing import List

from python.publishing.pattern_library.pattern_identity import build_canonical_mask_signature
from python.publishing.qc.validate_pattern import validate_pattern_record
from python.publishing.schemas.models import PatternRecord

_ALLOWED_PATTERN_STATUSES = {
    "active",
    "retired",
    "draft",
    "archived",
}


def validate_pattern_record_strict(pattern: PatternRecord) -> List[str]:
    """
    Strict validator for pattern library ingestion.

    This wraps the QC validator and adds pattern-library-specific
    constraints such as classic 9x9 expectations and metadata quality checks.
    """
    errors = validate_pattern_record(pattern)

    if pattern.grid_size != 9:
        errors.append("Wave 4 pattern ingestion currently supports only classic 9x9 assets")

    if pattern.layout_type != "classic9x9":
        errors.append("Wave 4 pattern ingestion currently expects layout_type='classic9x9'")

    if pattern.clue_count < 17:
        errors.append("clue_count must be >= 17 for a meaningful classic Sudoku clue pattern")

    if pattern.clue_count > 81:
        errors.append("clue_count must be <= 81")

    if not pattern.visual_family.strip():
        errors.append("visual_family must not be blank")

    if not pattern.family_id or not str(pattern.family_id).strip():
        errors.append("family_id must not be blank")

    if not pattern.family_name or not str(pattern.family_name).strip():
        errors.append("family_name must not be blank")

    if pattern.status not in _ALLOWED_PATTERN_STATUSES:
        errors.append(f"Unsupported pattern status: {pattern.status}")

    try:
        expected_signature = build_canonical_mask_signature(pattern.mask81)
        if pattern.canonical_mask_signature != expected_signature:
            errors.append(
                "canonical_mask_signature mismatch: stored signature does not match mask81 canonical signature"
            )
    except Exception as exc:
        errors.append(f"Failed to compute canonical_mask_signature: {exc}")

    return errors