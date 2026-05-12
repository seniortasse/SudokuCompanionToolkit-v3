from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from python.publishing.cleanup.delete_models import DeleteAction
from python.publishing.inventory.assignment_ledger import get_inventory_entry


def _legacy_inventory_only_check(
    inventory: Dict,
    *,
    record_id: str,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    entry = get_inventory_entry(inventory, record_id=record_id)

    if entry is None:
        return True, reasons

    if int(entry.get("assignment_count", 0)) > 0:
        reasons.append(
            f"record_id {record_id} is still assigned to {entry.get('assignment_count')} book placement(s)"
        )

    candidate_status = str(entry.get("candidate_status", "")).strip().lower()
    if candidate_status == "assigned":
        reasons.append(f"record_id {record_id} currently has candidate_status='assigned'")

    return len(reasons) == 0, reasons


def can_remove_record_from_catalog(
    inventory: Dict,
    *,
    record_id: str,
    library_id: str | None = None,
    records_dir: Path = Path("datasets/sudoku_books/classic9/puzzle_records"),
    inventory_dir: Path = Path("datasets/sudoku_books/classic9/catalogs"),
    books_dir: Path = Path("datasets/sudoku_books/classic9/books"),
    publications_dir: Path = Path("datasets/sudoku_books/classic9/publications"),
    requested_action: str = DeleteAction.DELETE,
) -> Tuple[bool, List[str]]:
    """
    Production-safe removal guard.

    If library_id is supplied, use the cleanup analyzer + policy engine so the
    decision is based on:
      - catalog index state
      - inventory state
      - built book references
      - publication references

    If library_id is omitted, fall back to the legacy inventory-only guard so
    older internal callers do not break unexpectedly.
    """
    if not library_id:
        return _legacy_inventory_only_check(inventory, record_id=record_id)
    

    from python.publishing.cleanup.dependency_analyzer import analyze_record_delete
    from python.publishing.cleanup.delete_policy import decide_record_delete

    plan = analyze_record_delete(
        record_id=record_id,
        library_id=library_id,
        requested_action=requested_action,
        records_dir=records_dir,
        inventory_dir=inventory_dir,
        books_dir=books_dir,
        publications_dir=publications_dir,
    )
    decision = decide_record_delete(plan)

    reasons: List[str] = []

    if decision.outcome == "BLOCK":
        reasons.append(decision.summary)
        for blocker in decision.blockers:
            reason = f"[{blocker.code}] {blocker.message}"
            if blocker.path:
                reason += f" ({blocker.path})"
            reasons.append(reason)
        return False, reasons

    if requested_action != decision.action:
        reasons.append(
            f"Requested action '{requested_action}' is not allowed; safest allowed action is '{decision.action}'."
        )
        for warning in decision.warnings:
            reasons.append(warning)
        return False, reasons

    return True, reasons