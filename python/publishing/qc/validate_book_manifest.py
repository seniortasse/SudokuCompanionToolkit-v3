from __future__ import annotations

from typing import List

from python.publishing.ids.validators import (
    is_valid_aisle_id,
    is_valid_book_id,
    is_valid_library_id,
    is_valid_section_id,
)
from python.publishing.schemas.models import BookManifest


_ALLOWED_PUZZLES_PER_PAGE = {2, 4, 6, 12}


def validate_book_manifest(book: BookManifest) -> List[str]:
    errors: List[str] = []

    if not is_valid_book_id(book.book_id):
        errors.append(f"Invalid book_id: {book.book_id}")

    if not is_valid_library_id(book.library_id):
        errors.append(f"Invalid library_id: {book.library_id}")

    if not is_valid_aisle_id(book.aisle_id):
        errors.append(f"Invalid aisle_id: {book.aisle_id}")

    if book.title.strip() == "":
        errors.append("title must not be blank")

    if book.trim_size.strip() == "":
        errors.append("trim_size must not be blank")

    if book.page_layout_profile.strip() == "":
        errors.append("page_layout_profile must not be blank")

    if book.grid_size <= 0:
        errors.append("grid_size must be > 0")

    if book.puzzles_per_page not in _ALLOWED_PUZZLES_PER_PAGE:
        errors.append(
            f"puzzles_per_page must be one of {_ALLOWED_PUZZLES_PER_PAGE}, got {book.puzzles_per_page}"
        )

    if book.puzzle_count < 0:
        errors.append("puzzle_count must be >= 0")

    if book.puzzle_count > 0 and book.puzzle_count % book.puzzles_per_page != 0:
        errors.append(
            f"puzzle_count ({book.puzzle_count}) must be divisible by puzzles_per_page ({book.puzzles_per_page})"
        )

    seen = set()
    for section_id in book.section_ids:
        if not is_valid_section_id(section_id):
            errors.append(f"Invalid section_id in section_ids: {section_id}")
        if section_id in seen:
            errors.append(f"Duplicate section_id in section_ids: {section_id}")
        seen.add(section_id)

    return errors