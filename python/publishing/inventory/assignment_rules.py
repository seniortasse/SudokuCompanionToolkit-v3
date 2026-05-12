from __future__ import annotations

from typing import Iterable, List

from python.publishing.inventory.assignment_ledger import is_record_assigned_in_library
from python.publishing.schemas.models import PuzzleRecord


def filter_records_available_for_library(
    *,
    records: Iterable[PuzzleRecord],
    inventory: dict,
) -> List[PuzzleRecord]:
    out: List[PuzzleRecord] = []
    for record in records:
        if record.candidate_status in {"archived", "rejected", "duplicate_blocked"}:
            continue
        if is_record_assigned_in_library(inventory, record_id=record.record_id):
            continue
        out.append(record)
    return out