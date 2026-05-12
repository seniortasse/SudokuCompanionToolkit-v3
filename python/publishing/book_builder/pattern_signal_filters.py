from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from python.publishing.schemas.models import PatternRecord, PuzzleRecord


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def record_matches_pattern_signals(
    record: PuzzleRecord,
    criteria: Dict[str, Any],
    pattern_records_by_id: Optional[Mapping[str, PatternRecord]],
) -> bool:
    if not criteria:
        return True

    needs_pattern_lookup = any(
        criteria.get(key)
        for key in [
            "required_pattern_tags_any",
            "excluded_pattern_tags",
            "min_pattern_generation_attempts",
            "min_pattern_success_rate",
            "min_pattern_unique_rate",
            "min_pattern_human_solvable_rate",
        ]
    )

    if not needs_pattern_lookup:
        return True

    if pattern_records_by_id is None:
        return False

    pattern_id = getattr(record, "pattern_id", None)
    if not pattern_id:
        return False

    pattern = pattern_records_by_id.get(str(pattern_id))
    if pattern is None:
        return False

    pattern_tags = {str(tag) for tag in list(pattern.tags or [])}
    required_pattern_tags_any = {str(x) for x in list(criteria.get("required_pattern_tags_any", []))}
    excluded_pattern_tags = {str(x) for x in list(criteria.get("excluded_pattern_tags", []))}

    if required_pattern_tags_any and not pattern_tags.intersection(required_pattern_tags_any):
        return False

    if excluded_pattern_tags and pattern_tags.intersection(excluded_pattern_tags):
        return False

    stats = dict(pattern.production_stats or {})

    min_attempts = _safe_int(criteria.get("min_pattern_generation_attempts"))
    if min_attempts is not None:
        attempts = _safe_int(stats.get("generation_attempts")) or 0
        if attempts < min_attempts:
            return False

    min_success_rate = _safe_float(criteria.get("min_pattern_success_rate"))
    if min_success_rate is not None:
        success_rate = _safe_float(stats.get("success_rate"))
        if success_rate is None or success_rate < min_success_rate:
            return False

    min_unique_rate = _safe_float(criteria.get("min_pattern_unique_rate"))
    if min_unique_rate is not None:
        unique_rate = _safe_float(stats.get("unique_rate"))
        if unique_rate is None or unique_rate < min_unique_rate:
            return False

    min_human_rate = _safe_float(criteria.get("min_pattern_human_solvable_rate"))
    if min_human_rate is not None:
        human_rate = _safe_float(stats.get("human_solvable_rate"))
        if human_rate is None or human_rate < min_human_rate:
            return False

    return True