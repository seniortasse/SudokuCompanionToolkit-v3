from __future__ import annotations

from typing import Dict, Iterable, List

from python.publishing.schemas.models import PuzzleRecord


def order_section_puzzles(
    puzzle_records: Iterable[PuzzleRecord],
    *,
    ordering_policy: Dict[str, object] | None = None,
) -> List[PuzzleRecord]:
    policy = dict(ordering_policy or {})
    within_section = str(policy.get("within_section", "weight_asc"))
    tie_breakers = list(policy.get("tie_breakers", ["technique_count", "clue_count", "generation_seed"]))

    records = list(puzzle_records)

    def sort_key(record: PuzzleRecord):
        primary = record.weight if within_section == "weight_asc" else record.weight

        tie_values = []
        for field_name in tie_breakers:
            value = getattr(record, field_name, None)
            if value is None:
                if field_name == "generation_seed":
                    value = 10**12
                else:
                    value = 10**9
            tie_values.append(value)

        return (primary, *tie_values, record.record_id)

    return sorted(records, key=sort_key)