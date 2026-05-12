from __future__ import annotations

from typing import Dict

from python.publishing.distribution.metadata_models import IsbnAssignment


_ISBN_ASSIGNMENTS: Dict[str, IsbnAssignment] = {
    "demo_empty": IsbnAssignment(
        isbn13="",
        isbn10="",
        assignment_name="demo_empty",
        status="unassigned",
    ),
}


def get_isbn_assignment(name: str) -> IsbnAssignment:
    try:
        return _ISBN_ASSIGNMENTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ISBN_ASSIGNMENTS.keys()))
        raise KeyError(f"Unknown ISBN assignment '{name}'. Known assignments: {known}") from exc