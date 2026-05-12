from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from python.publishing.schemas.models import PuzzleRecord


def _record_pattern_key(record: PuzzleRecord) -> str:
    return str(record.pattern_id or "")


def _record_family_key(record: PuzzleRecord) -> str:
    value = getattr(record, "pattern_family_id", None)
    return str(value or "")


def apply_distribution_constraints(
    *,
    records: Iterable[PuzzleRecord],
    target_count: int,
    min_distinct_patterns: int | None,
    max_distinct_patterns: int | None,
    min_distinct_pattern_families: int | None,
    max_distinct_pattern_families: int | None,
    pattern_occurrence_caps: Dict[str, int] | None,
    pattern_family_occurrence_caps: Dict[str, int] | None,
) -> List[PuzzleRecord]:
    selected: List[PuzzleRecord] = []
    pattern_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    pattern_caps = {str(k): int(v) for k, v in dict(pattern_occurrence_caps or {}).items()}
    family_caps = {str(k): int(v) for k, v in dict(pattern_family_occurrence_caps or {}).items()}

    for record in records:
        if len(selected) >= target_count:
            break

        pattern_key = _record_pattern_key(record)
        family_key = _record_family_key(record)

        pattern_cap = pattern_caps.get(pattern_key)
        if pattern_cap is not None and pattern_counts[pattern_key] >= pattern_cap:
            continue

        family_cap = family_caps.get(family_key)
        if family_cap is not None and family_counts[family_key] >= family_cap:
            continue

        if max_distinct_patterns is not None and pattern_key:
            current_distinct_patterns = len({key for key, count in pattern_counts.items() if count > 0})
            adding_new_pattern = pattern_counts[pattern_key] == 0
            if adding_new_pattern and current_distinct_patterns >= int(max_distinct_patterns):
                continue

        if max_distinct_pattern_families is not None and family_key:
            current_distinct_families = len({key for key, count in family_counts.items() if count > 0})
            adding_new_family = family_counts[family_key] == 0
            if adding_new_family and current_distinct_families >= int(max_distinct_pattern_families):
                continue

        selected.append(record)
        if pattern_key:
            pattern_counts[pattern_key] += 1
        if family_key:
            family_counts[family_key] += 1

    if min_distinct_patterns is not None:
        distinct_patterns = len({(_record_pattern_key(r)) for r in selected if _record_pattern_key(r)})
        if distinct_patterns < int(min_distinct_patterns):
            return []

    if min_distinct_pattern_families is not None:
        distinct_families = len({(_record_family_key(r)) for r in selected if _record_family_key(r)})
        if distinct_families < int(min_distinct_pattern_families):
            return []

    return selected