from __future__ import annotations

from pathlib import Path


def get_publication_dir(
    *,
    book_id: str,
    publication_id: str,
    base_dir: Path,
) -> Path:
    book_token = str(book_id or "").strip()
    publication_token = str(publication_id or "").strip()

    if not book_token:
        raise ValueError("book_id is required")
    if not publication_token:
        raise ValueError("publication_id is required")

    folder_name = f"{book_token}__{publication_token}"
    return Path(base_dir) / folder_name