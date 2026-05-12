from __future__ import annotations

from typing import List

from python.publishing.ids.validators import is_valid_library_id
from python.publishing.qc.validate_book_manifest import validate_book_manifest
from python.publishing.schemas.models import CatalogManifest


def validate_catalog_manifest(catalog: CatalogManifest) -> List[str]:
    errors: List[str] = []

    if catalog.catalog_version.strip() == "":
        errors.append("catalog_version must not be blank")

    if catalog.generated_at.strip() == "":
        errors.append("generated_at must not be blank")

    seen_library_ids = set()
    for library_id in catalog.library_ids:
        if not is_valid_library_id(library_id):
            errors.append(f"Invalid library_id in library_ids: {library_id}")
        if library_id in seen_library_ids:
            errors.append(f"Duplicate library_id in library_ids: {library_id}")
        seen_library_ids.add(library_id)

    manifest_library_ids = {library.library_id for library in catalog.libraries}
    if set(catalog.library_ids) != manifest_library_ids:
        errors.append(
            "library_ids must exactly match the set of library.library_id values in libraries"
        )

    for summary in catalog.book_summaries:
        if summary.book_id.strip() == "":
            errors.append("book_summary.book_id must not be blank")
        if summary.title.strip() == "":
            errors.append(f"book_summary.title must not be blank for {summary.book_id}")
        if summary.aisle_id.strip() == "":
            errors.append(f"book_summary.aisle_id must not be blank for {summary.book_id}")
        if summary.puzzle_count < 0:
            errors.append(f"book_summary.puzzle_count must be >= 0 for {summary.book_id}")

    for key, value in catalog.index_files.items():
        if str(key).strip() == "":
            errors.append("index_files keys must not be blank")
        if str(value).strip() == "":
            errors.append(f"index_files[{key}] must not be blank")

    return errors