from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from python.publishing.schemas.models import PuzzleRecord


def _default_record_entry(*, record_id: str, candidate_status: str) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "candidate_status": candidate_status,
        "assignment_count": 0,
        "assignments": [],
        "last_book_id": None,
        "last_section_id": None,
        "last_puzzle_uid": None,
    }


def get_inventory_entry(
    inventory: Dict[str, Any],
    *,
    record_id: str,
) -> Optional[Dict[str, Any]]:
    return inventory.get("records", {}).get(record_id)


def is_record_assigned_in_library(
    inventory: Dict[str, Any],
    *,
    record_id: str,
) -> bool:
    entry = get_inventory_entry(inventory, record_id=record_id)
    if entry is None:
        return False
    return int(entry.get("assignment_count", 0)) > 0


def list_assigned_record_ids(inventory: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for record_id, entry in dict(inventory.get("records", {})).items():
        if int(entry.get("assignment_count", 0)) > 0:
            out.add(str(record_id))
    return out


def ensure_inventory_entry(
    inventory: Dict[str, Any],
    *,
    record: PuzzleRecord,
) -> Dict[str, Any]:
    records = inventory.setdefault("records", {})
    entry = records.get(record.record_id)
    if entry is None:
        entry = _default_record_entry(
            record_id=record.record_id,
            candidate_status=record.candidate_status,
        )
        records[record.record_id] = entry
    return entry


def register_assignment(
    inventory: Dict[str, Any],
    *,
    record: PuzzleRecord,
) -> None:
    if not record.book_id or not record.section_id:
        raise ValueError("Cannot register assignment without book_id and section_id")

    entry = ensure_inventory_entry(inventory, record=record)

    assignment = {
        "book_id": record.book_id,
        "section_id": record.section_id,
        "section_code": record.section_code,
        "puzzle_uid": record.puzzle_uid,
        "local_puzzle_code": record.local_puzzle_code,
        "position_in_section": record.position_in_section,
        "position_in_book": record.position_in_book,
    }

    assignments = list(entry.get("assignments", []))
    for existing in assignments:
        if (
            existing.get("book_id") == assignment["book_id"]
            and existing.get("section_id") == assignment["section_id"]
            and existing.get("puzzle_uid") == assignment["puzzle_uid"]
        ):
            entry["candidate_status"] = "assigned"
            entry["assignment_count"] = len(assignments)
            entry["last_book_id"] = record.book_id
            entry["last_section_id"] = record.section_id
            entry["last_puzzle_uid"] = record.puzzle_uid
            return

    assignments.append(assignment)
    entry["assignments"] = assignments
    entry["assignment_count"] = len(assignments)
    entry["candidate_status"] = "assigned"
    entry["last_book_id"] = record.book_id
    entry["last_section_id"] = record.section_id
    entry["last_puzzle_uid"] = record.puzzle_uid


def _refresh_entry_after_assignment_change(entry: Dict[str, Any]) -> None:
    kept = list(entry.get("assignments", []))
    entry["assignment_count"] = len(kept)

    if kept:
        last = kept[-1]
        entry["candidate_status"] = "assigned"
        entry["last_book_id"] = last.get("book_id")
        entry["last_section_id"] = last.get("section_id")
        entry["last_puzzle_uid"] = last.get("puzzle_uid")
    else:
        entry["candidate_status"] = "available"
        entry["last_book_id"] = None
        entry["last_section_id"] = None
        entry["last_puzzle_uid"] = None


def unregister_assignments_for_book(
    inventory: Dict[str, Any],
    *,
    book_id: str,
) -> List[str]:
    affected: List[str] = []

    for record_id, entry in dict(inventory.get("records", {})).items():
        assignments = list(entry.get("assignments", []))
        kept = [a for a in assignments if a.get("book_id") != book_id]

        if len(kept) != len(assignments):
            affected.append(str(record_id))
            entry["assignments"] = kept
            _refresh_entry_after_assignment_change(entry)

    return affected


def unregister_assignment_for_record(
    inventory: Dict[str, Any],
    *,
    record_id: str,
    book_id: str,
    section_id: str | None = None,
    puzzle_uid: str | None = None,
) -> bool:
    entry = get_inventory_entry(inventory, record_id=record_id)
    if entry is None:
        return False

    assignments = list(entry.get("assignments", []))
    kept = []
    removed = False

    for assignment in assignments:
        same_book = assignment.get("book_id") == book_id
        same_section = section_id is None or assignment.get("section_id") == section_id
        same_uid = puzzle_uid is None or assignment.get("puzzle_uid") == puzzle_uid

        if same_book and same_section and same_uid:
            removed = True
            continue

        kept.append(assignment)

    if removed:
        entry["assignments"] = kept
        _refresh_entry_after_assignment_change(entry)

    return removed


def sync_catalog_statuses_from_inventory(
    *,
    records: Iterable[PuzzleRecord],
    inventory: Dict[str, Any],
) -> None:
    for record in records:
        entry = get_inventory_entry(inventory, record_id=record.record_id)
        if entry is None:
            continue
        if int(entry.get("assignment_count", 0)) > 0:
            record.candidate_status = "assigned"
        elif record.candidate_status == "assigned":
            record.candidate_status = "available"