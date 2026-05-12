from __future__ import annotations

from typing import Any, Dict

from python.publishing.schemas.models import PuzzleRecord
from python.publishing.techniques.technique_catalog import normalize_technique_id


def _normalize_technique_name(value: Any) -> str:
    return normalize_technique_id(str(value))


def _matches_histogram_ranges(
    record: PuzzleRecord,
    ranges: Dict[str, Dict[str, int]],
) -> bool:
    normalized_histogram = {
        _normalize_technique_name(k): int(v)
        for k, v in dict(record.technique_histogram or {}).items()
    }

    for technique_name, bounds in dict(ranges or {}).items():
        norm_name = _normalize_technique_name(technique_name)
        count = int(normalized_histogram.get(norm_name, 0))
        min_value = bounds.get("min")
        max_value = bounds.get("max")

        if min_value is not None and count < int(min_value):
            return False
        if max_value is not None and count > int(max_value):
            return False

    return True


def record_matches_global_filters(
    record: PuzzleRecord,
    global_filters: Dict[str, Any],
) -> bool:
    if not global_filters:
        return True

    layout_type = global_filters.get("layout_type")
    if layout_type is not None and record.layout_type != str(layout_type):
        return False

    is_unique = global_filters.get("is_unique")
    if is_unique is not None and record.is_unique != bool(is_unique):
        return False

    is_human_solvable = global_filters.get("is_human_solvable")
    if is_human_solvable is not None and record.is_human_solvable != bool(is_human_solvable):
        return False

    candidate_status_in = list(global_filters.get("candidate_status_in", []))
    if candidate_status_in and record.candidate_status not in candidate_status_in:
        return False

    return True


def record_matches_section_criteria(
    record: PuzzleRecord,
    criteria: Dict[str, Any],
) -> bool:
    if not criteria:
        return True

    weight_min = criteria.get("weight_min")
    if weight_min is not None and record.weight < int(weight_min):
        return False

    weight_max = criteria.get("weight_max")
    if weight_max is not None and record.weight > int(weight_max):
        return False

    clue_count_min = criteria.get("clue_count_min")
    if clue_count_min is not None and record.clue_count < int(clue_count_min):
        return False

    clue_count_max = criteria.get("clue_count_max")
    if clue_count_max is not None and record.clue_count > int(clue_count_max):
        return False

    technique_count_min = criteria.get("technique_count_min")
    if technique_count_min is not None and record.technique_count < int(technique_count_min):
        return False

    technique_count_max = criteria.get("technique_count_max")
    if technique_count_max is not None and record.technique_count > int(technique_count_max):
        return False

    puzzle_difficulty = criteria.get("puzzle_difficulty")
    if puzzle_difficulty is not None and record.puzzle_difficulty != str(puzzle_difficulty):
        return False

    puzzle_difficulty_in = list(criteria.get("puzzle_difficulty_in", []))
    if puzzle_difficulty_in and record.puzzle_difficulty not in {str(x) for x in puzzle_difficulty_in}:
        return False

    used = {_normalize_technique_name(t) for t in set(record.techniques_used)}

    required_techniques = [_normalize_technique_name(t) for t in list(criteria.get("required_techniques", []))]
    if required_techniques and not all(t in used for t in required_techniques):
        return False

    required_any_techniques = [_normalize_technique_name(t) for t in list(criteria.get("required_any_techniques", []))]
    if required_any_techniques and not any(t in used for t in required_any_techniques):
        return False

    excluded_techniques = [_normalize_technique_name(t) for t in list(criteria.get("excluded_techniques", []))]
    if excluded_techniques and any(t in used for t in excluded_techniques):
        return False

    featured_techniques = {_normalize_technique_name(x) for x in list(criteria.get("featured_techniques", []))}
    featured = _normalize_technique_name(record.featured_technique) if record.featured_technique else None
    if featured_techniques and featured not in featured_techniques:
        return False

    pattern_ids = list(criteria.get("pattern_ids", []))
    if pattern_ids and record.pattern_id not in {str(x) for x in pattern_ids}:
        return False

    pattern_names = list(criteria.get("pattern_names", []))
    if pattern_names and record.pattern_name not in {str(x) for x in pattern_names}:
        return False

    pattern_family_ids = list(criteria.get("pattern_family_ids", []))
    if pattern_family_ids:
        record_family_id = getattr(record, "pattern_family_id", None)
        if record_family_id not in {str(x) for x in pattern_family_ids}:
            return False

    excluded_pattern_ids = list(criteria.get("excluded_pattern_ids", []))
    if excluded_pattern_ids and record.pattern_id in {str(x) for x in excluded_pattern_ids}:
        return False

    technique_histogram_ranges = dict(criteria.get("technique_histogram_ranges", {}))
    if technique_histogram_ranges and not _matches_histogram_ranges(record, technique_histogram_ranges):
        return False

    return True