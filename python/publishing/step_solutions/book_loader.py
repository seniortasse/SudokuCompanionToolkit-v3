from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from python.publishing.step_solutions.models import (
    StepSolutionBookInfo,
    StepSolutionPackageRequest,
    StepSolutionPuzzleRecord,
)


BOOK_MANIFEST_FILENAME = "book_manifest.json"
PUZZLES_DIRNAME = "puzzles"


def read_json(path: Path) -> Dict[str, Any]:
    """
    Read a UTF-8 JSON object from disk.
    """

    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_book_dir(
    book_id: str,
    books_root: Path = Path("datasets/sudoku_books/classic9/books"),
) -> Path:
    """
    Resolve the folder for one generated book.

    Example:
        datasets/sudoku_books/classic9/books/BK-CL9-DW-B01
    """

    return Path(books_root) / book_id


def resolve_book_manifest_path(
    book_id: str,
    books_root: Path = Path("datasets/sudoku_books/classic9/books"),
) -> Path:
    return resolve_book_dir(book_id, books_root) / BOOK_MANIFEST_FILENAME


def resolve_book_puzzles_dir(
    book_id: str,
    books_root: Path = Path("datasets/sudoku_books/classic9/books"),
) -> Path:
    return resolve_book_dir(book_id, books_root) / PUZZLES_DIRNAME


def load_book_info(
    book_id: str,
    books_root: Path = Path("datasets/sudoku_books/classic9/books"),
) -> StepSolutionBookInfo:
    """
    Load minimal book metadata from book_manifest.json.
    """

    book_dir = resolve_book_dir(book_id, books_root)
    manifest_path = book_dir / BOOK_MANIFEST_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Book manifest not found for {book_id!r}: {manifest_path}"
        )

    raw = read_json(manifest_path)

    manifest_book_id = str(raw.get("book_id") or "").strip()
    if manifest_book_id and manifest_book_id != book_id:
        raise ValueError(
            f"Book id mismatch. Requested {book_id!r}, "
            f"but manifest contains {manifest_book_id!r}."
        )

    return StepSolutionBookInfo(
        book_id=book_id,
        title=str(raw.get("title") or ""),
        subtitle=str(raw.get("subtitle") or ""),
        puzzle_count=int(raw.get("puzzle_count") or 0),
        grid_size=int(raw.get("grid_size") or 9),
        section_ids=list(raw.get("section_ids") or []),
        manifest_path=manifest_path,
        book_dir=book_dir,
        raw=raw,
    )


def load_book_puzzle_records(
    book_id: str,
    books_root: Path = Path("datasets/sudoku_books/classic9/books"),
) -> List[StepSolutionPuzzleRecord]:
    """
    Load all puzzle records from a book's puzzles/ folder.

    Records are sorted by:
        1. position_in_book
        2. section_code
        3. position_in_section
        4. local_puzzle_code
        5. record_id
    """

    puzzles_dir = resolve_book_puzzles_dir(book_id, books_root)

    if not puzzles_dir.exists():
        raise FileNotFoundError(
            f"Puzzles folder not found for {book_id!r}: {puzzles_dir}"
        )

    records: List[StepSolutionPuzzleRecord] = []

    for path in sorted(puzzles_dir.glob("*.json")):
        raw = read_json(path)
        record = normalize_puzzle_record(raw=raw, source_path=path, fallback_book_id=book_id)
        records.append(record)

    records.sort(
        key=lambda record: (
            _sort_int(record.position_in_book),
            record.section_code,
            _sort_int(record.position_in_section),
            record.local_puzzle_code,
            record.record_id,
        )
    )

    return records


def normalize_puzzle_record(
    raw: Dict[str, Any],
    source_path: Path,
    fallback_book_id: str,
) -> StepSolutionPuzzleRecord:
    """
    Normalize one puzzle JSON object into the stable step-solution record shape.
    """

    record_id = str(raw.get("record_id") or source_path.stem)
    book_id = str(raw.get("book_id") or fallback_book_id)
    section_id = str(raw.get("section_id") or "")
    section_code = str(raw.get("section_code") or "")
    local_puzzle_code = str(raw.get("local_puzzle_code") or "").strip()
    friendly_puzzle_id = str(raw.get("friendly_puzzle_id") or "")

    if not local_puzzle_code:
        raise ValueError(
            f"Puzzle record {source_path} is missing local_puzzle_code."
        )

    givens81 = str(raw.get("givens81") or "").strip()
    solution81 = str(raw.get("solution81") or "").strip()

    validate_grid81(givens81, field_name="givens81", source_path=source_path)
    validate_grid81(solution81, field_name="solution81", source_path=source_path)

    return StepSolutionPuzzleRecord(
        record_id=record_id,
        book_id=book_id,
        section_id=section_id,
        section_code=section_code,
        local_puzzle_code=local_puzzle_code,
        friendly_puzzle_id=friendly_puzzle_id,
        givens81=givens81,
        solution81=solution81,
        position_in_book=int(raw.get("position_in_book") or 0),
        position_in_section=int(raw.get("position_in_section") or 0),
        difficulty_label=str(raw.get("difficulty_label") or ""),
        puzzle_difficulty=str(raw.get("puzzle_difficulty") or ""),
        weight=_optional_int(raw.get("weight")),
        technique_count=_optional_int(raw.get("technique_count")),
        techniques_used=list(raw.get("techniques_used") or []),
        technique_histogram=dict(raw.get("technique_histogram") or {}),
        source_path=source_path,
        raw=raw,
    )


def validate_grid81(value: str, field_name: str, source_path: Path) -> None:
    """
    Validate a classic 9x9 flattened grid string.
    """

    if len(value) != 81:
        raise ValueError(
            f"{source_path} has invalid {field_name}: expected 81 characters, "
            f"got {len(value)}."
        )

    allowed = set("0123456789.")
    invalid = sorted(set(value) - allowed)
    if invalid:
        raise ValueError(
            f"{source_path} has invalid {field_name}: unexpected characters "
            f"{invalid!r}."
        )


def select_puzzle_records(
    records: Iterable[StepSolutionPuzzleRecord],
    only_puzzle: Optional[str] = None,
    only_section: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[StepSolutionPuzzleRecord]:
    """
    Select a subset of puzzle records for a dry run or export.

    only_puzzle accepts either:
        - internal code: L1-001
        - external code: L-1-1

    only_section accepts:
        - L1
        - L-1
        - SEC-L1
    """

    selected = list(records)

    if only_puzzle:
        selected = [
            record
            for record in selected
            if _matches_puzzle_selector(record, only_puzzle)
        ]

    if only_section:
        selected = [
            record
            for record in selected
            if _matches_section_selector(record, only_section)
        ]

    if limit is not None:
        selected = selected[: int(limit)]

    return selected


def load_selected_book_puzzles(
    request: StepSolutionPackageRequest,
    only_puzzle: Optional[str] = None,
    only_section: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[StepSolutionPuzzleRecord]:
    """
    Convenience loader using StepSolutionPackageRequest paths.
    """

    all_records = load_book_puzzle_records(
        book_id=request.book_id,
        books_root=request.books_root,
    )
    return select_puzzle_records(
        records=all_records,
        only_puzzle=only_puzzle,
        only_section=only_section,
        limit=limit,
    )


def _matches_puzzle_selector(record: StepSolutionPuzzleRecord, selector: str) -> bool:
    value = str(selector or "").strip()
    if not value:
        return True

    if record.local_puzzle_code == value:
        return True

    # Avoid importing identity at module import time to keep this loader simple.
    from python.publishing.step_solutions.identity import internal_to_external_puzzle_code

    try:
        return internal_to_external_puzzle_code(record.local_puzzle_code) == value
    except ValueError:
        return False


def _matches_section_selector(record: StepSolutionPuzzleRecord, selector: str) -> bool:
    value = str(selector or "").strip()
    if not value:
        return True

    normalized = value.replace("-", "")

    if record.section_code == value:
        return True
    if record.section_id == value:
        return True
    if record.section_code.replace("-", "") == normalized:
        return True
    if record.section_id.replace("-", "") == normalized:
        return True

    return False


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _sort_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0