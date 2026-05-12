from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# IMPORTANT:
# Do not import book_package_store at module import time.
# It pulls in book_builder -> inventory -> removal_guard -> cleanup again,
# which creates a circular import during repair/audit flows.
from python.publishing.cleanup.delete_models import (
    DeleteAction,
    DeleteBlocker,
    DeleteDependency,
    DeletePlan,
    DeleteTarget,
)
from python.publishing.inventory.assignment_ledger import get_inventory_entry
from python.publishing.inventory.library_inventory_store import load_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index


DEFAULT_BOOKS_DIR = Path("datasets/sudoku_books/classic9/books")
DEFAULT_BOOK_SPECS_DIR = Path("datasets/sudoku_books/classic9/book_specs")
DEFAULT_PUBLICATIONS_DIR = Path("datasets/sudoku_books/classic9/publications")
DEFAULT_PUBLICATION_SPECS_DIR = Path("datasets/sudoku_books/classic9/publication_specs")
DEFAULT_RECORDS_DIR = Path("datasets/sudoku_books/classic9/puzzle_records")
DEFAULT_INVENTORY_DIR = Path("datasets/sudoku_books/classic9/catalogs")
DEFAULT_CANDIDATES_JSONL = Path("runs/publishing/classic9/puzzle_generation/candidates.jsonl")


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _iter_book_dirs(books_dir: Path) -> Iterable[Path]:
    if not books_dir.exists():
        return []
    return sorted(p for p in books_dir.iterdir() if p.is_dir())


def _iter_publication_dirs(publications_dir: Path) -> Iterable[Path]:
    if not publications_dir.exists():
        return []
    return sorted(p for p in publications_dir.iterdir() if p.is_dir())


def _find_book_spec_paths(book_specs_dir: Path, *, book_id: str) -> List[Path]:
    if not book_specs_dir.exists():
        return []
    return sorted(book_specs_dir.glob(f"{book_id}*.json"))


def _find_publication_spec_paths(publication_specs_dir: Path, *, book_id: str) -> List[Path]:
    if not publication_specs_dir.exists():
        return []
    return sorted(publication_specs_dir.glob(f"{book_id}*.json"))


def _scan_books_for_record(books_dir: Path, *, record_id: str) -> List[DeleteDependency]:
    from python.publishing.book_builder.book_package_store import load_built_book_package

    hits: List[DeleteDependency] = []

    for book_dir in _iter_book_dirs(books_dir):
        try:
            book_manifest, _sections, assigned_puzzles = load_built_book_package(book_dir)
        except Exception:
            hits.append(
                DeleteDependency(
                    dependency_type="book_scan_error",
                    path=_normalize_path(book_dir),
                    detail="Could not fully scan this book directory",
                    blocks_hard_delete=True,
                    blocks_archive=False,
                )
            )
            continue

        for record in assigned_puzzles:
            if str(record.record_id) != record_id:
                continue

            detail = (
                f"record_id {record_id} appears in built book {book_manifest.book_id}"
            )
            hits.append(
                DeleteDependency(
                    dependency_type="book_assignment",
                    path=_normalize_path(book_dir / "puzzles" / f"{record.puzzle_uid or record.record_id}.json"),
                    detail=detail,
                    reference_id=book_manifest.book_id,
                    blocks_hard_delete=True,
                    blocks_archive=True,
                )
            )

    return hits


def _scan_publications_for_book(publications_dir: Path, *, book_id: str) -> List[DeleteDependency]:
    hits: List[DeleteDependency] = []

    for publication_dir in _iter_publication_dirs(publications_dir):
        manifest_path = publication_dir / "publication_manifest.json"
        package_path = publication_dir / "publication_package.json"

        manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None
        package = _safe_load_json(package_path) if package_path.exists() else None

        manifest_book_id = str((manifest or {}).get("book_id", "")).strip()
        package_book_id = str((package or {}).get("book_id", "")).strip()

        if manifest_book_id == book_id or package_book_id == book_id or publication_dir.name.startswith(f"{book_id}__"):
            pub_id = str((manifest or {}).get("publication_id") or (package or {}).get("publication_id") or publication_dir.name)
            hits.append(
                DeleteDependency(
                    dependency_type="publication_package",
                    path=_normalize_path(publication_dir),
                    detail=f"publication package exists for book_id {book_id}",
                    reference_id=pub_id,
                    blocks_hard_delete=True,
                    blocks_archive=False,
                )
            )

    return hits


def _scan_publications_for_record(publications_dir: Path, *, record_id: str) -> List[DeleteDependency]:
    hits: List[DeleteDependency] = []

    for publication_dir in _iter_publication_dirs(publications_dir):
        package_path = publication_dir / "publication_package.json"
        manifest_path = publication_dir / "publication_manifest.json"

        package = _safe_load_json(package_path) if package_path.exists() else None
        manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None

        package_text = json.dumps(package, ensure_ascii=False) if package is not None else ""
        manifest_text = json.dumps(manifest, ensure_ascii=False) if manifest is not None else ""

        if record_id in package_text or record_id in manifest_text:
            ref_id = str((manifest or {}).get("publication_id") or publication_dir.name)
            hits.append(
                DeleteDependency(
                    dependency_type="publication_reference",
                    path=_normalize_path(publication_dir),
                    detail=f"record_id {record_id} was found in publication metadata",
                    reference_id=ref_id,
                    blocks_hard_delete=True,
                    blocks_archive=True,
                )
            )

    return hits



def analyze_publication_delete(
    *,
    publication_id: str,
    requested_action: str = DeleteAction.DELETE,
    publications_dir: Path = DEFAULT_PUBLICATIONS_DIR,
    publication_specs_dir: Path = DEFAULT_PUBLICATION_SPECS_DIR,
) -> DeletePlan:
    target = DeleteTarget(
        target_type="publication",
        target_id=publication_id,
        display_name=publication_id,
    )
    plan = DeletePlan(
        target=target,
        requested_action=requested_action,
        allowed_actions=[DeleteAction.DELETE, DeleteAction.CASCADE_DELETE],
    )

    publication_dir = publications_dir / publication_id
    plan.metadata["publication_dir"] = _normalize_path(publication_dir)

    if publication_dir.exists():
        plan.files_to_delete.append(_normalize_path(publication_dir))
    else:
        plan.blockers.append(
            DeleteBlocker(
                code="PUBLICATION_DIR_MISSING",
                message=f"Publication directory not found for publication_id {publication_id}",
                path=_normalize_path(publication_dir),
                target_id=publication_id,
            )
        )

    manifest_path = publication_dir / "publication_manifest.json"
    package_path = publication_dir / "publication_package.json"

    manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None
    package = _safe_load_json(package_path) if package_path.exists() else None

    if manifest is not None:
        plan.metadata["publication_manifest"] = manifest
    if package is not None:
        plan.metadata["publication_package"] = package

    derived_book_id = str(
        (manifest or {}).get("book_id")
        or (package or {}).get("book_id")
        or ""
    ).strip()
    if derived_book_id:
        plan.metadata["book_id"] = derived_book_id
        plan.notes.append(f"Publication appears linked to book_id {derived_book_id}.")

    spec_hits: List[Path] = []
    if publication_specs_dir.exists():
        spec_hits.extend(sorted(publication_specs_dir.glob(f"{publication_id}*.json")))
        if derived_book_id:
            spec_hits.extend(sorted(publication_specs_dir.glob(f"{derived_book_id}*.json")))

    seen_spec_paths = set()
    for spec_path in spec_hits:
        norm_path = _normalize_path(spec_path)
        if norm_path in seen_spec_paths:
            continue
        seen_spec_paths.add(norm_path)

        plan.dependencies.append(
            DeleteDependency(
                dependency_type="publication_spec",
                path=norm_path,
                detail=f"Publication spec exists for publication_id {publication_id}",
                reference_id=publication_id,
                blocks_hard_delete=False,
                blocks_archive=False,
            )
        )

    if publication_dir.exists():
        plan.notes.append("Publication directory exists.")
    else:
        plan.notes.append("Publication directory is missing.")

    if not plan.dependencies:
        plan.notes.append("No publication spec dependencies were found.")
    else:
        plan.notes.append("At least one publication spec dependency was found.")

    return plan




def analyze_record_delete(
    *,
    record_id: str,
    library_id: str,
    requested_action: str = DeleteAction.DELETE,
    records_dir: Path = DEFAULT_RECORDS_DIR,
    inventory_dir: Path = DEFAULT_INVENTORY_DIR,
    books_dir: Path = DEFAULT_BOOKS_DIR,
    publications_dir: Path = DEFAULT_PUBLICATIONS_DIR,
) -> DeletePlan:
    target = DeleteTarget(
        target_type="puzzle_record",
        target_id=record_id,
        display_name=record_id,
    )
    plan = DeletePlan(
        target=target,
        requested_action=requested_action,
        allowed_actions=[DeleteAction.ARCHIVE, DeleteAction.DELETE],
    )

    record_path = records_dir / f"{record_id}.json"
    plan.metadata["record_path"] = _normalize_path(record_path)
    plan.files_to_update.append(_normalize_path(records_dir / "_catalog_index.json"))

    if record_path.exists():
        plan.files_to_delete.append(_normalize_path(record_path))
    else:
        plan.blockers.append(
            DeleteBlocker(
                code="RECORD_FILE_MISSING",
                message=f"Record file not found for record_id {record_id}",
                path=_normalize_path(record_path),
                target_id=record_id,
            )
        )

    catalog_index = load_catalog_index(records_dir)
    index_entry = dict(catalog_index.get("records_by_id", {}).get(record_id, {}))
    if not index_entry:
        plan.blockers.append(
            DeleteBlocker(
                code="CATALOG_INDEX_MISSING_ENTRY",
                message=f"record_id {record_id} was not found in _catalog_index.json",
                path=_normalize_path(records_dir / "_catalog_index.json"),
                target_id=record_id,
            )
        )
    else:
        plan.metadata["catalog_index_entry"] = index_entry
        candidate_status = str(index_entry.get("candidate_status", "")).strip().lower()
        if candidate_status:
            plan.metadata["catalog_candidate_status"] = candidate_status

    inventory = load_library_inventory(base_dir=inventory_dir, library_id=library_id)
    inventory_entry = get_inventory_entry(inventory, record_id=record_id)
    if inventory_entry is not None:
        assignment_count = int(inventory_entry.get("assignment_count", 0))
        plan.metadata["inventory_entry"] = dict(inventory_entry)

        if assignment_count > 0:
            plan.dependencies.append(
                DeleteDependency(
                    dependency_type="inventory_assignment",
                    path=_normalize_path(inventory_dir / "_library_inventory.json"),
                    detail=f"record_id {record_id} still has {assignment_count} assignment(s) in inventory",
                    reference_id=record_id,
                    blocks_hard_delete=True,
                    blocks_archive=True,
                )
            )

        candidate_status = str(inventory_entry.get("candidate_status", "")).strip().lower()
        if candidate_status == "assigned":
            plan.dependencies.append(
                DeleteDependency(
                    dependency_type="inventory_status",
                    path=_normalize_path(inventory_dir / "_library_inventory.json"),
                    detail=f"record_id {record_id} has inventory candidate_status='assigned'",
                    reference_id=record_id,
                    blocks_hard_delete=True,
                    blocks_archive=True,
                )
            )

    plan.dependencies.extend(_scan_books_for_record(books_dir, record_id=record_id))
    plan.dependencies.extend(_scan_publications_for_record(publications_dir, record_id=record_id))

    if record_path.exists():
        plan.notes.append("Canonical record file exists.")
    else:
        plan.notes.append("Canonical record file is missing.")

    if not plan.dependencies:
        plan.notes.append("No downstream book or publication dependencies were found.")
    else:
        plan.notes.append("At least one downstream dependency was found.")

    return plan


def analyze_book_delete(
    *,
    book_id: str,
    library_id: str,
    requested_action: str = DeleteAction.DELETE,
    books_dir: Path = DEFAULT_BOOKS_DIR,
    book_specs_dir: Path = DEFAULT_BOOK_SPECS_DIR,
    publications_dir: Path = DEFAULT_PUBLICATIONS_DIR,
    publication_specs_dir: Path = DEFAULT_PUBLICATION_SPECS_DIR,
    inventory_dir: Path = DEFAULT_INVENTORY_DIR,
    records_dir: Path = DEFAULT_RECORDS_DIR,
) -> DeletePlan:
    from python.publishing.book_builder.book_package_store import load_built_book_package

    target = DeleteTarget(
        target_type="book",
        target_id=book_id,
        display_name=book_id,
    )
    plan = DeletePlan(
        target=target,
        requested_action=requested_action,
        allowed_actions=[DeleteAction.DELETE, DeleteAction.CASCADE_DELETE],
    )

    book_dir = books_dir / book_id
    plan.metadata["book_dir"] = _normalize_path(book_dir)

    if book_dir.exists():
        plan.files_to_delete.append(_normalize_path(book_dir))
    else:
        plan.blockers.append(
            DeleteBlocker(
                code="BOOK_DIR_MISSING",
                message=f"Book directory not found for book_id {book_id}",
                path=_normalize_path(book_dir),
                target_id=book_id,
            )
        )

    inventory = load_library_inventory(base_dir=inventory_dir, library_id=library_id)
    inventory_records = dict(inventory.get("records", {}))

    freed_record_ids: List[str] = []
    for record_id, entry in inventory_records.items():
        assignments = list(entry.get("assignments", []))
        for assignment in assignments:
            if str(assignment.get("book_id", "")).strip() == book_id:
                freed_record_ids.append(str(record_id))
                break

    if freed_record_ids:
        plan.files_to_update.append(_normalize_path(inventory_dir / "_library_inventory.json"))
        plan.files_to_update.append(_normalize_path(records_dir / "_catalog_index.json"))
        plan.metadata["freed_record_ids"] = sorted(set(freed_record_ids))

    for dep in _scan_publications_for_book(publications_dir, book_id=book_id):
        plan.dependencies.append(dep)

    for path in _find_book_spec_paths(book_specs_dir, book_id=book_id):
        plan.dependencies.append(
            DeleteDependency(
                dependency_type="book_spec",
                path=_normalize_path(path),
                detail=f"Book spec exists for book_id {book_id}",
                reference_id=book_id,
                blocks_hard_delete=False,
                blocks_archive=False,
            )
        )

    for path in _find_publication_spec_paths(publication_specs_dir, book_id=book_id):
        plan.dependencies.append(
            DeleteDependency(
                dependency_type="publication_spec",
                path=_normalize_path(path),
                detail=f"Publication spec exists for book_id {book_id}",
                reference_id=book_id,
                blocks_hard_delete=True,
                blocks_archive=False,
            )
        )

    if book_dir.exists():
        try:
            book_manifest, _sections, assigned_puzzles = load_built_book_package(book_dir)
            plan.metadata["book_manifest"] = book_manifest.to_dict()
            plan.metadata["book_puzzle_count"] = len(assigned_puzzles)
            plan.metadata["book_assigned_record_ids"] = sorted({str(item.record_id) for item in assigned_puzzles})
        except Exception:
            plan.blockers.append(
                DeleteBlocker(
                    code="BOOK_SCAN_FAILED",
                    message=f"Failed to fully load built book package for {book_id}",
                    path=_normalize_path(book_dir),
                    target_id=book_id,
                )
            )

    if not plan.dependencies:
        plan.notes.append("No downstream publication or spec dependencies were found.")
    else:
        plan.notes.append("At least one downstream publication/spec dependency was found.")

    return plan


def analyze_candidate_jsonl_delete(
    *,
    jsonl_path: Path = DEFAULT_CANDIDATES_JSONL,
    requested_action: str = DeleteAction.RAW_DELETE,
    line_numbers: Optional[List[int]] = None,
    generation_seeds: Optional[List[int]] = None,
    pattern_ids: Optional[List[str]] = None,
    givens81_values: Optional[List[str]] = None,
    solution81_values: Optional[List[str]] = None,
    records_dir: Path = DEFAULT_RECORDS_DIR,
) -> DeletePlan:
    target = DeleteTarget(
        target_type="candidate_jsonl",
        target_id=str(jsonl_path),
        display_name=jsonl_path.name,
    )
    plan = DeletePlan(
        target=target,
        requested_action=requested_action,
        allowed_actions=[DeleteAction.RAW_DELETE],
    )

    plan.metadata["jsonl_path"] = _normalize_path(jsonl_path)

    if not jsonl_path.exists():
        plan.blockers.append(
            DeleteBlocker(
                code="CANDIDATES_JSONL_MISSING",
                message=f"Candidates JSONL file not found: {jsonl_path}",
                path=_normalize_path(jsonl_path),
                target_id=str(jsonl_path),
            )
        )
        return plan

    selected_lines: List[Dict[str, Any]] = []
    raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    normalized_pattern_ids = {str(x).strip() for x in (pattern_ids or []) if str(x).strip()}
    normalized_givens = {str(x).strip() for x in (givens81_values or []) if str(x).strip()}
    normalized_solutions = {str(x).strip() for x in (solution81_values or []) if str(x).strip()}
    normalized_line_numbers = {int(x) for x in (line_numbers or [])}
    normalized_generation_seeds = {int(x) for x in (generation_seeds or [])}

    for index, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            payload = json.loads(stripped)
        except Exception:
            plan.dependencies.append(
                DeleteDependency(
                    dependency_type="jsonl_parse_warning",
                    path=_normalize_path(jsonl_path),
                    detail=f"Line {index} could not be parsed as JSON and will be ignored by selector logic",
                    reference_id=str(index),
                    blocks_hard_delete=False,
                    blocks_archive=False,
                )
            )
            continue

        match = False
        if normalized_line_numbers and index in normalized_line_numbers:
            match = True
        if normalized_generation_seeds and int(payload.get("generation_seed", -1)) in normalized_generation_seeds:
            match = True
        if normalized_pattern_ids and str(payload.get("pattern_id", "")).strip() in normalized_pattern_ids:
            match = True
        if normalized_givens and str(payload.get("givens81", "")).strip() in normalized_givens:
            match = True
        if normalized_solutions and str(payload.get("solution81", "")).strip() in normalized_solutions:
            match = True

        if match:
            payload["_line_number"] = index
            selected_lines.append(payload)

    plan.metadata["selected_count"] = len(selected_lines)
    plan.metadata["selected_lines"] = selected_lines[:50]
    plan.files_to_update.append(_normalize_path(jsonl_path))

    catalog_index = load_catalog_index(records_dir)
    signature_to_record_id = dict(catalog_index.get("solution_signature_to_record_id", {}))

    for item in selected_lines:
        solution81 = str(item.get("solution81", "")).strip()
        record_id = signature_to_record_id.get(solution81)
        if record_id:
            plan.dependencies.append(
                DeleteDependency(
                    dependency_type="canonical_record_match",
                    path=_normalize_path(records_dir / f"{record_id}.json"),
                    detail=f"Selected raw candidate matches canonical record_id {record_id} by solution signature",
                    reference_id=record_id,
                    blocks_hard_delete=False,
                    blocks_archive=False,
                )
            )

    if not selected_lines:
        plan.warnings.append("Selector matched zero JSONL candidate lines.")

    return plan



def analyze_publication_spec_delete(
    *,
    publication_spec_id: str,
    requested_action: str = DeleteAction.DELETE,
    publication_specs_dir: Path = DEFAULT_PUBLICATION_SPECS_DIR,
) -> DeletePlan:
    target = DeleteTarget(
        target_type="publication_spec",
        target_id=publication_spec_id,
        display_name=publication_spec_id,
    )
    plan = DeletePlan(
        target=target,
        requested_action=requested_action,
        allowed_actions=[DeleteAction.DELETE],
    )

    candidate_paths = sorted(publication_specs_dir.glob(f"{publication_spec_id}*.json"))
    if not candidate_paths:
        plan.blockers.append(
            DeleteBlocker(
                code="PUBLICATION_SPEC_MISSING",
                message=f"No publication spec files matched publication_spec_id {publication_spec_id}",
                path=_normalize_path(publication_specs_dir),
                target_id=publication_spec_id,
            )
        )
        return plan

    for spec_path in candidate_paths:
        plan.files_to_delete.append(_normalize_path(spec_path))

    plan.metadata["matched_spec_count"] = len(candidate_paths)
    plan.metadata["matched_spec_paths"] = [_normalize_path(p) for p in candidate_paths]
    plan.notes.append(f"Matched {len(candidate_paths)} publication spec file(s).")
    return plan