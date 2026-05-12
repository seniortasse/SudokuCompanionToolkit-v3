from __future__ import annotations

from typing import Iterable, List

from python.publishing.step_solutions.identity import make_step_solution_identity
from python.publishing.step_solutions.models import (
    StepSolutionPuzzleInstance,
    StepSolutionPuzzleRecord,
)


def puzzle_record_to_instance(
    record: StepSolutionPuzzleRecord,
) -> StepSolutionPuzzleInstance:
    """
    Convert one normalized book puzzle record into the stable instance shape
    used by later log-generation phases.
    """

    identity = make_step_solution_identity(
        book_id=record.book_id,
        local_puzzle_code=record.local_puzzle_code,
    )

    return StepSolutionPuzzleInstance(
        book_id=identity.book_id,
        internal_puzzle_code=identity.internal_puzzle_code,
        external_puzzle_code=identity.external_puzzle_code,
        commercial_book_code=identity.commercial_book_code,
        commercial_problem_id=identity.commercial_problem_id,
        givens81=record.givens81,
        solution81=record.solution81,
        grid_size=9,
        section_code=record.section_code,
        position_in_book=record.position_in_book,
        position_in_section=record.position_in_section,
        difficulty_label=record.difficulty_label,
        weight=record.weight,
        source_record_id=record.record_id,
        source_path=record.source_path,
    )


def puzzle_records_to_instances(
    records: Iterable[StepSolutionPuzzleRecord],
) -> List[StepSolutionPuzzleInstance]:
    """
    Convert many normalized book puzzle records into step-solution instances.
    """

    return [puzzle_record_to_instance(record) for record in records]


def givens81_to_grid(givens81: str) -> List[List[int]]:
    """
    Convert a flattened givens81 string into a 9x9 integer grid.

    Empty cells represented by 0 or . become 0.
    """

    return _grid81_to_int_grid(givens81)


def solution81_to_grid(solution81: str) -> List[List[int]]:
    """
    Convert a flattened solution81 string into a 9x9 integer grid.
    """

    return _grid81_to_int_grid(solution81)


def instance_to_legacy_payload(instance: StepSolutionPuzzleInstance) -> dict:
    """
    Build a conservative legacy-style payload for later integration with the
    old step-by-step solution generator.

    This phase only defines the adapter shape. Phase 4 will wire this into the
    actual user-log generator.
    """

    return {
        "book_id": instance.book_id,
        "book_code": instance.commercial_book_code,
        "internal_puzzle_code": instance.internal_puzzle_code,
        "external_puzzle_code": instance.external_puzzle_code,
        "problem_id": instance.commercial_problem_id,
        "givens81": instance.givens81,
        "solution81": instance.solution81,
        "givens_grid": givens81_to_grid(instance.givens81),
        "solution_grid": solution81_to_grid(instance.solution81),
        "section_code": instance.section_code,
        "position_in_book": instance.position_in_book,
        "position_in_section": instance.position_in_section,
        "difficulty_label": instance.difficulty_label,
        "weight": instance.weight,
        "source_record_id": instance.source_record_id,
        "source_path": str(instance.source_path) if instance.source_path else "",
    }


def _grid81_to_int_grid(value: str) -> List[List[int]]:
    cleaned = str(value or "").strip()
    if len(cleaned) != 81:
        raise ValueError(f"Expected 81 characters, got {len(cleaned)}.")

    cells: List[int] = []
    for char in cleaned:
        if char in ("0", "."):
            cells.append(0)
        elif char in "123456789":
            cells.append(int(char))
        else:
            raise ValueError(f"Unexpected grid character: {char!r}")

    return [cells[index : index + 9] for index in range(0, 81, 9)]