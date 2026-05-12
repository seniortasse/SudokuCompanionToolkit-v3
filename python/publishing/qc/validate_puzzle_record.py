from __future__ import annotations

from typing import List

from python.publishing.ids.validators import (
    is_valid_aisle_id,
    is_valid_book_id,
    is_valid_friendly_puzzle_id,
    is_valid_library_id,
    is_valid_local_puzzle_code,
    is_valid_pattern_id,
    is_valid_puzzle_uid,
    is_valid_record_id,
    is_valid_section_id,
)
from python.publishing.puzzle_catalog.catalog_identity import ALL_CANDIDATE_STATUSES
from python.publishing.puzzle_catalog.solution_signature import build_solution_signature
from python.publishing.schemas.models import PuzzleRecord

_ALLOWED_PUZZLE_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
    "expert",
    "genius",
}

_ALLOWED_TECHNIQUE_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
    "very_hard",
    "expert",
    "genius",
}


def _is_81_grid(value: str) -> bool:
    return len(value) == 81


def validate_puzzle_record(puzzle: PuzzleRecord) -> List[str]:
    errors: List[str] = []

    if not is_valid_record_id(puzzle.record_id):
        errors.append(f"Invalid record_id: {puzzle.record_id}")

    if puzzle.candidate_status not in ALL_CANDIDATE_STATUSES:
        errors.append(f"Invalid candidate_status: {puzzle.candidate_status}")

    if not puzzle.solution_signature or len(puzzle.solution_signature) != 81:
        errors.append("solution_signature must be exactly 81 characters long")

    if not is_valid_library_id(puzzle.library_id):
        errors.append(f"Invalid library_id: {puzzle.library_id}")

    if puzzle.aisle_id is not None and not is_valid_aisle_id(puzzle.aisle_id):
        errors.append(f"Invalid aisle_id: {puzzle.aisle_id}")

    if puzzle.book_id is not None and not is_valid_book_id(puzzle.book_id):
        errors.append(f"Invalid book_id: {puzzle.book_id}")

    if puzzle.section_id is not None and not is_valid_section_id(puzzle.section_id):
        errors.append(f"Invalid section_id: {puzzle.section_id}")

    if puzzle.local_puzzle_code is not None and not is_valid_local_puzzle_code(puzzle.local_puzzle_code):
        errors.append(f"Invalid local_puzzle_code: {puzzle.local_puzzle_code}")

    if puzzle.friendly_puzzle_id is not None and not is_valid_friendly_puzzle_id(puzzle.friendly_puzzle_id):
        errors.append(f"Invalid friendly_puzzle_id: {puzzle.friendly_puzzle_id}")

    if puzzle.puzzle_uid is not None and not is_valid_puzzle_uid(puzzle.puzzle_uid):
        errors.append(f"Invalid puzzle_uid: {puzzle.puzzle_uid}")

    if puzzle.pattern_id is not None and not is_valid_pattern_id(puzzle.pattern_id):
        errors.append(f"Invalid pattern_id: {puzzle.pattern_id}")

    if puzzle.pattern_family_id is not None and not str(puzzle.pattern_family_id).strip():
        errors.append("pattern_family_id must not be blank when provided")

    if puzzle.pattern_family_name is not None and not str(puzzle.pattern_family_name).strip():
        errors.append("pattern_family_name must not be blank when provided")

    if puzzle.grid_size <= 0:
        errors.append("grid_size must be > 0")

    if not _is_81_grid(puzzle.givens81):
        errors.append("givens81 must be exactly 81 characters long")

    if not _is_81_grid(puzzle.solution81):
        errors.append("solution81 must be exactly 81 characters long")

    if len(puzzle.charset) != puzzle.grid_size:
        errors.append(
            f"charset length must match grid_size: len(charset)={len(puzzle.charset)} grid_size={puzzle.grid_size}"
        )

    computed_clues = sum(1 for ch in puzzle.givens81 if ch != "0")
    if puzzle.clue_count != computed_clues:
        errors.append(
            f"clue_count mismatch: declared={puzzle.clue_count}, computed={computed_clues}"
        )

    if not puzzle.is_unique:
        errors.append("Puzzle records must have is_unique=True")

    if not puzzle.is_human_solvable:
        errors.append("Puzzle records must have is_human_solvable=True")

    if puzzle.weight < 0:
        errors.append("weight must be >= 0")

    if puzzle.technique_count < 0:
        errors.append("technique_count must be >= 0")

    if puzzle.technique_count != len(set(puzzle.techniques_used)):
        errors.append(
            "technique_count should equal the number of distinct entries in techniques_used"
        )

    if puzzle.puzzle_difficulty not in _ALLOWED_PUZZLE_DIFFICULTIES:
        errors.append(f"Invalid puzzle_difficulty: {puzzle.puzzle_difficulty}")

    for item in puzzle.techniques_difficulty:
        if item not in _ALLOWED_TECHNIQUE_DIFFICULTIES:
            errors.append(f"Invalid techniques_difficulty entry: {item}")

    try:
        expected_signature = build_solution_signature(puzzle.solution81)
        if puzzle.solution_signature != expected_signature:
            errors.append(
                "solution_signature mismatch: stored signature does not match solution81 canonical signature"
            )
    except Exception as exc:
        errors.append(f"Failed to compute canonical solution_signature: {exc}")

    if puzzle.print_header.display_code.strip() == "":
        errors.append("print_header.display_code must not be blank")

    return errors