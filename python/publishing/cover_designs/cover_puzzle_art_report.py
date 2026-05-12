from __future__ import annotations

from typing import Any


def _is_valid_givens81(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    cleaned = value.strip()
    if len(cleaned) != 81:
        return False

    return set(cleaned).issubset(set("0123456789.-"))


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    requested_source = dict(selection.get("requested_source") or {})
    return {
        "role": selection.get("role"),
        "requested_source": requested_source,
        "fallback_used": bool(selection.get("fallback_used")),
        "fallback_reason": selection.get("fallback_reason"),
        "record_id": selection.get("record_id"),
        "pattern_id": selection.get("pattern_id"),
        "position_in_book": selection.get("position_in_book"),
        "page": selection.get("page"),
        "puzzle_on_page": selection.get("puzzle_on_page"),
        "local_puzzle_code": selection.get("local_puzzle_code"),
        "givens81": selection.get("givens81"),
        "givens81_valid": _is_valid_givens81(selection.get("givens81")),
    }


def build_cover_puzzle_art_report(resolved_puzzle_art: dict[str, Any] | None) -> dict[str, Any]:
    """
    Build a validation/debug report for cover puzzle-art selection.

    This report is intentionally separate from the resolver:
    - resolver chooses puzzles
    - report explains whether the chosen data is usable
    """
    if not resolved_puzzle_art:
        return {
            "status": "not_resolved",
            "errors": [],
            "warnings": [
                "No resolved_puzzle_art found. Run generate_cover_front_art with --book-dir to resolve book-driven cover grids."
            ],
            "book_record_count": 0,
            "selections": {},
        }

    errors: list[str] = []
    warnings: list[str] = []

    selections_raw = dict(resolved_puzzle_art.get("selections") or {})
    selections: dict[str, Any] = {}

    for role in ("main", "left", "right"):
        raw = dict(selections_raw.get(role) or {})
        summary = _selection_summary(raw)
        selections[role] = summary

        if not summary["givens81_valid"]:
            errors.append(f"{role}: selected givens81 is missing or invalid.")

        if summary["fallback_used"]:
            warnings.append(
                f"{role}: fallback used"
                + (f" — {summary['fallback_reason']}" if summary.get("fallback_reason") else "")
            )

        requested = dict(summary.get("requested_source") or {})
        if requested.get("mode") == "pattern_id" and summary["fallback_used"]:
            warnings.append(
                f"{role}: requested pattern_id {requested.get('pattern_id')} was not found in the book."
            )

        if requested.get("mode") == "specific_record_id" and summary["fallback_used"]:
            warnings.append(
                f"{role}: requested record_id {requested.get('record_id')} was not found or had invalid givens81."
            )

        if requested.get("mode") == "book_index" and summary["fallback_used"]:
            warnings.append(
                f"{role}: requested book_index {requested.get('index')} was out of range or invalid."
            )

    book_record_count = int(resolved_puzzle_art.get("book_record_count") or 0)
    if book_record_count <= 0:
        warnings.append("No book puzzle records were available to the puzzle-art resolver.")

    status = "ok" if not errors else "failed"
    if not errors and warnings:
        status = "ok_with_warnings"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "book_record_count": book_record_count,
        "book_dir": resolved_puzzle_art.get("book_dir"),
        "puzzle_records_dir": resolved_puzzle_art.get("puzzle_records_dir"),
        "selections": selections,
    }