from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from python.publishing.book_builder.criteria_engine import (
    record_matches_global_filters,
    record_matches_section_criteria,
)
from python.publishing.book_builder.pattern_signal_filters import record_matches_pattern_signals
from python.publishing.schemas.models import PatternRecord, PuzzleRecord


def select_puzzles_for_section(
    *,
    puzzle_records: Iterable[PuzzleRecord],
    section_criteria: Dict[str, Any],
    global_filters: Dict[str, Any],
    pattern_records_by_id: Optional[Mapping[str, PatternRecord]] = None,
) -> List[PuzzleRecord]:
    selected: List[PuzzleRecord] = []

    for record in puzzle_records:
        if not record_matches_global_filters(record, global_filters):
            continue
        if not record_matches_section_criteria(record, section_criteria):
            continue
        if not record_matches_pattern_signals(record, section_criteria, pattern_records_by_id):
            continue
        selected.append(record)

    return selected