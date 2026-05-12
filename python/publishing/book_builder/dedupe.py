from __future__ import annotations

from typing import Iterable, List, Set

from python.publishing.schemas.models import PuzzleRecord


def build_reuse_blocklist(
    *,
    already_used_puzzle_ids: Iterable[str],
) -> Set[str]:
    return {str(value) for value in already_used_puzzle_ids if str(value).strip()}


def filter_reusable_puzzles(
    puzzle_records: Iterable[PuzzleRecord],
    *,
    blocklist: Set[str],
    reuse_policy: str,
) -> List[PuzzleRecord]:
    records = list(puzzle_records)

    if reuse_policy == "allow_reuse":
        return records

    return [record for record in records if record.record_id not in blocklist]