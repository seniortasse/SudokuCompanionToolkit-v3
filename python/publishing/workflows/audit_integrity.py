from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from python.publishing.book_builder.book_package_store import load_built_book_package
from python.publishing.inventory.library_inventory_store import load_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_record_files(records_dir: Path) -> List[Path]:
    if not records_dir.exists():
        return []

    candidate_paths = sorted(
        p for p in records_dir.glob("*.json")
        if p.name != "_catalog_index.json"
        and not p.name.startswith("_")
    )

    record_paths: List[Path] = []
    for path in candidate_paths:
        payload = _safe_load_json(path)
        if payload is None:
            # Keep invalid JSON files in scope so audit can report them.
            record_paths.append(path)
            continue

        embedded_record_id = str(payload.get("record_id", "")).strip()
        # Treat as a canonical record file only if it self-identifies as one.
        if embedded_record_id:
            record_paths.append(path)

    return record_paths


def _iter_book_dirs(books_dir: Path) -> List[Path]:
    if not books_dir.exists():
        return []
    return sorted(p for p in books_dir.iterdir() if p.is_dir())


def _iter_publication_dirs(publications_dir: Path) -> List[Path]:
    if not publications_dir.exists():
        return []
    return sorted(p for p in publications_dir.iterdir() if p.is_dir())


def _iter_json_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted(p for p in base_dir.glob("*.json") if p.is_file())


def _build_issue(
    *,
    severity: str,
    code: str,
    message: str,
    path: str | None = None,
    target_id: str | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if path:
        payload["path"] = path
    if target_id:
        payload["target_id"] = target_id
    if details:
        payload["details"] = dict(details)
    return payload


def audit_integrity(
    *,
    library_id: str,
    records_dir: Path,
    inventory_dir: Path,
    books_dir: Path,
    book_specs_dir: Path,
    publications_dir: Path,
    publication_specs_dir: Path,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    inventory = load_library_inventory(base_dir=inventory_dir, library_id=library_id)
    catalog_index = load_catalog_index(records_dir)

    inventory_records = dict(inventory.get("records", {}))
    index_records_by_id = dict(catalog_index.get("records_by_id", {}))
    solution_sig_to_record_id = dict(catalog_index.get("solution_signature_to_record_id", {}))

    record_files = _iter_record_files(records_dir)
    record_file_ids = {p.stem for p in record_files}

    index_record_ids = set(index_records_by_id.keys())
    inventory_record_ids = set(inventory_records.keys())

    # ------------------------------------------------------------------
    # 1) Catalog index <-> record file consistency
    # ------------------------------------------------------------------
    for record_id in sorted(record_file_ids - index_record_ids):
        issues.append(_build_issue(
            severity="error",
            code="RECORD_FILE_MISSING_FROM_INDEX",
            message=f"Record file exists but _catalog_index.json has no entry for record_id {record_id}",
            path=_normalize_path(records_dir / f"{record_id}.json"),
            target_id=record_id,
        ))

    for record_id in sorted(index_record_ids - record_file_ids):
        issues.append(_build_issue(
            severity="error",
            code="INDEX_ENTRY_MISSING_RECORD_FILE",
            message=f"_catalog_index.json contains record_id {record_id}, but the record file is missing",
            path=_normalize_path(records_dir / "_catalog_index.json"),
            target_id=record_id,
        ))

    for record_id in sorted(index_record_ids & record_file_ids):
        record_path = records_dir / f"{record_id}.json"
        payload = _safe_load_json(record_path)
        if payload is None:
            issues.append(_build_issue(
                severity="error",
                code="RECORD_FILE_INVALID_JSON",
                message=f"Record file could not be parsed as JSON for record_id {record_id}",
                path=_normalize_path(record_path),
                target_id=record_id,
            ))
            continue

        file_record_id = str(payload.get("record_id", "")).strip()
        if file_record_id and file_record_id != record_id:
            issues.append(_build_issue(
                severity="error",
                code="RECORD_ID_MISMATCH",
                message=f"Record file stem and embedded record_id disagree for {record_id}",
                path=_normalize_path(record_path),
                target_id=record_id,
                details={"embedded_record_id": file_record_id},
            ))

        solution81 = str(payload.get("solution81", "")).strip()
        if solution81:
            mapped_record_id = str(solution_sig_to_record_id.get(solution81, "")).strip()
            if mapped_record_id and mapped_record_id != record_id:
                issues.append(_build_issue(
                    severity="error",
                    code="SOLUTION_SIGNATURE_MAP_MISMATCH",
                    message=f"solution_signature_to_record_id points to {mapped_record_id}, but file belongs to {record_id}",
                    path=_normalize_path(records_dir / "_catalog_index.json"),
                    target_id=record_id,
                    details={"solution81": solution81, "mapped_record_id": mapped_record_id},
                ))

    # ------------------------------------------------------------------
    # 2) Inventory <-> index consistency
    # ------------------------------------------------------------------
    for record_id in sorted(inventory_record_ids - index_record_ids):
        issues.append(_build_issue(
            severity="error",
            code="INVENTORY_RECORD_MISSING_FROM_INDEX",
            message=f"Inventory contains record_id {record_id}, but _catalog_index.json does not",
            path=_normalize_path(inventory_dir / "_library_inventory.json"),
            target_id=record_id,
        ))

    for record_id in sorted(index_record_ids & inventory_record_ids):
        inventory_entry = dict(inventory_records.get(record_id, {}))
        index_entry = dict(index_records_by_id.get(record_id, {}))

        inventory_status = str(inventory_entry.get("candidate_status", "")).strip().lower()
        index_status = str(index_entry.get("candidate_status", "")).strip().lower()

        if inventory_status and index_status and inventory_status != index_status:
            issues.append(_build_issue(
                severity="warning",
                code="STATUS_MISMATCH_INDEX_VS_INVENTORY",
                message=f"candidate_status mismatch for record_id {record_id}: inventory={inventory_status}, index={index_status}",
                path=_normalize_path(inventory_dir / "_library_inventory.json"),
                target_id=record_id,
                details={
                    "inventory_status": inventory_status,
                    "index_status": index_status,
                },
            ))

    # ------------------------------------------------------------------
    # 3) Books <-> inventory consistency
    # ------------------------------------------------------------------
    book_dir_names = {p.name for p in _iter_book_dirs(books_dir)}
    book_assigned_record_ids: Dict[str, List[str]] = {}
    all_book_record_ids: set[str] = set()

    for book_dir in _iter_book_dirs(books_dir):
        try:
            book_manifest, _sections, assigned_puzzles = load_built_book_package(book_dir)
        except Exception as exc:
            issues.append(_build_issue(
                severity="error",
                code="BOOK_SCAN_FAILED",
                message=f"Failed to fully load built book package for {book_dir.name}",
                path=_normalize_path(book_dir),
                target_id=book_dir.name,
                details={"error": str(exc)},
            ))
            continue

        manifest_book_id = str(getattr(book_manifest, "book_id", "")).strip()
        if manifest_book_id and manifest_book_id != book_dir.name:
            issues.append(_build_issue(
                severity="error",
                code="BOOK_DIR_NAME_MISMATCH",
                message=f"Book directory name and manifest book_id disagree for {book_dir.name}",
                path=_normalize_path(book_dir),
                target_id=book_dir.name,
                details={"manifest_book_id": manifest_book_id},
            ))

        book_assigned_record_ids[book_dir.name] = []
        for assigned in assigned_puzzles:
            record_id = str(getattr(assigned, "record_id", "")).strip()
            if not record_id:
                continue

            book_assigned_record_ids[book_dir.name].append(record_id)
            all_book_record_ids.add(record_id)

            if record_id not in record_file_ids:
                issues.append(_build_issue(
                    severity="error",
                    code="BOOK_REFERENCES_MISSING_RECORD_FILE",
                    message=f"Built book {book_dir.name} references missing canonical record_id {record_id}",
                    path=_normalize_path(book_dir),
                    target_id=record_id,
                ))

    for record_id, inventory_entry in sorted(inventory_records.items()):
        assignments = list(inventory_entry.get("assignments", []))
        assignment_count = int(inventory_entry.get("assignment_count", 0))

        if assignment_count != len(assignments):
            issues.append(_build_issue(
                severity="warning",
                code="ASSIGNMENT_COUNT_MISMATCH",
                message=f"assignment_count does not equal assignments length for record_id {record_id}",
                path=_normalize_path(inventory_dir / "_library_inventory.json"),
                target_id=record_id,
                details={
                    "assignment_count": assignment_count,
                    "assignments_len": len(assignments),
                },
            ))

        for assignment in assignments:
            book_id = str(assignment.get("book_id", "")).strip()
            if not book_id:
                issues.append(_build_issue(
                    severity="warning",
                    code="ASSIGNMENT_MISSING_BOOK_ID",
                    message=f"Inventory assignment is missing book_id for record_id {record_id}",
                    path=_normalize_path(inventory_dir / "_library_inventory.json"),
                    target_id=record_id,
                ))
                continue

            if book_id not in book_dir_names:
                issues.append(_build_issue(
                    severity="error",
                    code="INVENTORY_REFERENCES_MISSING_BOOK",
                    message=f"Inventory assignment points to missing book_id {book_id} for record_id {record_id}",
                    path=_normalize_path(inventory_dir / "_library_inventory.json"),
                    target_id=record_id,
                ))
                continue

            if record_id not in set(book_assigned_record_ids.get(book_id, [])):
                issues.append(_build_issue(
                    severity="error",
                    code="INVENTORY_ASSIGNMENT_NOT_IN_BOOK",
                    message=f"Inventory says record_id {record_id} is assigned to book_id {book_id}, but built book scan did not confirm it",
                    path=_normalize_path(inventory_dir / "_library_inventory.json"),
                    target_id=record_id,
                    details={"book_id": book_id},
                ))

    for book_id, record_ids in sorted(book_assigned_record_ids.items()):
        record_id_set = set(record_ids)
        for record_id in sorted(record_id_set):
            inventory_entry = inventory_records.get(record_id)
            if inventory_entry is None:
                issues.append(_build_issue(
                    severity="error",
                    code="BOOK_RECORD_MISSING_FROM_INVENTORY",
                    message=f"Built book {book_id} contains record_id {record_id}, but inventory has no entry",
                    path=_normalize_path(books_dir / book_id),
                    target_id=record_id,
                    details={"book_id": book_id},
                ))
                continue

            assignments = list(inventory_entry.get("assignments", []))
            if not any(str(a.get("book_id", "")).strip() == book_id for a in assignments):
                issues.append(_build_issue(
                    severity="error",
                    code="BOOK_RECORD_NOT_ASSIGNED_IN_INVENTORY",
                    message=f"Built book {book_id} contains record_id {record_id}, but inventory has no matching assignment",
                    path=_normalize_path(books_dir / book_id),
                    target_id=record_id,
                    details={"book_id": book_id},
                ))

    # ------------------------------------------------------------------
    # 4) Publication dirs/specs consistency
    # ------------------------------------------------------------------
    publication_dir_names = {p.name for p in _iter_publication_dirs(publications_dir)}
    referenced_book_ids_from_publications: set[str] = set()

    for publication_dir in _iter_publication_dirs(publications_dir):
        manifest_path = publication_dir / "publication_manifest.json"
        package_path = publication_dir / "publication_package.json"

        manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None
        package = _safe_load_json(package_path) if package_path.exists() else None

        if manifest is None and package is None:
            issues.append(_build_issue(
                severity="warning",
                code="PUBLICATION_METADATA_MISSING",
                message=f"Publication directory {publication_dir.name} contains neither a readable publication_manifest.json nor publication_package.json",
                path=_normalize_path(publication_dir),
                target_id=publication_dir.name,
            ))
            continue

        publication_id_manifest = str((manifest or {}).get("publication_id", "")).strip()
        publication_id_package = str((package or {}).get("publication_id", "")).strip()
        book_id = str((manifest or {}).get("book_id") or (package or {}).get("book_id") or "").strip()

        if publication_id_manifest and publication_id_manifest != publication_dir.name:
            issues.append(_build_issue(
                severity="warning",
                code="PUBLICATION_DIR_NAME_MISMATCH",
                message=f"Publication directory name and manifest publication_id disagree for {publication_dir.name}",
                path=_normalize_path(publication_dir),
                target_id=publication_dir.name,
                details={"manifest_publication_id": publication_id_manifest},
            ))

        if publication_id_package and publication_id_manifest and publication_id_package != publication_id_manifest:
            issues.append(_build_issue(
                severity="warning",
                code="PUBLICATION_ID_MISMATCH",
                message=f"publication_manifest.json and publication_package.json disagree on publication_id in {publication_dir.name}",
                path=_normalize_path(publication_dir),
                target_id=publication_dir.name,
                details={
                    "manifest_publication_id": publication_id_manifest,
                    "package_publication_id": publication_id_package,
                },
            ))

        if book_id:
            referenced_book_ids_from_publications.add(book_id)
            if book_id not in book_dir_names:
                issues.append(_build_issue(
                    severity="error",
                    code="PUBLICATION_REFERENCES_MISSING_BOOK",
                    message=f"Publication {publication_dir.name} references missing book_id {book_id}",
                    path=_normalize_path(publication_dir),
                    target_id=publication_dir.name,
                    details={"book_id": book_id},
                ))

    for spec_path in _iter_json_files(publication_specs_dir):
        spec_payload = _safe_load_json(spec_path)
        if spec_payload is None:
            issues.append(_build_issue(
                severity="warning",
                code="PUBLICATION_SPEC_INVALID_JSON",
                message=f"Publication spec file could not be parsed as JSON: {spec_path.name}",
                path=_normalize_path(spec_path),
                target_id=spec_path.stem,
            ))
            continue

        book_id = str(spec_payload.get("book_id", "")).strip()
        if book_id and book_id not in book_dir_names:
            issues.append(_build_issue(
                severity="warning",
                code="PUBLICATION_SPEC_REFERENCES_MISSING_BOOK",
                message=f"Publication spec {spec_path.name} references missing book_id {book_id}",
                path=_normalize_path(spec_path),
                target_id=spec_path.stem,
                details={"book_id": book_id},
            ))

    # ------------------------------------------------------------------
    # 5) Book spec consistency
    # ------------------------------------------------------------------
    for spec_path in _iter_json_files(book_specs_dir):
        spec_payload = _safe_load_json(spec_path)
        if spec_payload is None:
            issues.append(_build_issue(
                severity="warning",
                code="BOOK_SPEC_INVALID_JSON",
                message=f"Book spec file could not be parsed as JSON: {spec_path.name}",
                path=_normalize_path(spec_path),
                target_id=spec_path.stem,
            ))
            continue

        book_id = str(spec_payload.get("book_id", "")).strip()
        if book_id and book_id not in book_dir_names:
            issues.append(_build_issue(
                severity="warning",
                code="BOOK_SPEC_REFERENCES_MISSING_BOOK",
                message=f"Book spec {spec_path.name} references missing book_id {book_id}",
                path=_normalize_path(spec_path),
                target_id=spec_path.stem,
                details={"book_id": book_id},
            ))

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")

    return {
        "generated_utc": _utc_stamp(),
        "library_id": library_id,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
            "issue_count": len(issues),
            "record_file_count": len(record_files),
            "index_record_count": len(index_record_ids),
            "inventory_record_count": len(inventory_record_ids),
            "book_dir_count": len(book_dir_names),
            "publication_dir_count": len(publication_dir_names),
        },
        "issues": issues,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit publishing platform integrity across records, inventory, books, publications, and specs."
    )
    parser.add_argument("--library-id", required=True)
    parser.add_argument(
        "--records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
    )
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
    )
    parser.add_argument(
        "--books-dir",
        default="datasets/sudoku_books/classic9/books",
    )
    parser.add_argument(
        "--book-specs-dir",
        default="datasets/sudoku_books/classic9/book_specs",
    )
    parser.add_argument(
        "--publications-dir",
        default="datasets/sudoku_books/classic9/publications",
    )
    parser.add_argument(
        "--publication-specs-dir",
        default="datasets/sudoku_books/classic9/publication_specs",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/integrity_reports",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    report = audit_integrity(
        library_id=args.library_id,
        records_dir=Path(args.records_dir),
        inventory_dir=Path(args.inventory_dir),
        books_dir=Path(args.books_dir),
        book_specs_dir=Path(args.book_specs_dir),
        publications_dir=Path(args.publications_dir),
        publication_specs_dir=Path(args.publication_specs_dir),
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{_utc_stamp()}__integrity_audit__{args.library_id}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report["summary"]
    print("=" * 72, flush=True)
    print("INTEGRITY AUDIT", flush=True)
    print("=" * 72, flush=True)
    print(f"Library id:       {args.library_id}", flush=True)
    print(f"Errors:           {summary['error_count']}", flush=True)
    print(f"Warnings:         {summary['warning_count']}", flush=True)
    print(f"Total issues:     {summary['issue_count']}", flush=True)
    print(f"Record files:     {summary['record_file_count']}", flush=True)
    print(f"Index records:    {summary['index_record_count']}", flush=True)
    print(f"Inventory records:{summary['inventory_record_count']}", flush=True)
    print(f"Book dirs:        {summary['book_dir_count']}", flush=True)
    print(f"Publication dirs: {summary['publication_dir_count']}", flush=True)
    print(f"Report path:      {report_path}", flush=True)

    if report["issues"]:
        print("", flush=True)
        print("Top issues:", flush=True)
        for issue in report["issues"][:20]:
            location = f" ({issue['path']})" if "path" in issue else ""
            print(
                f"- [{issue['severity']}] {issue['code']}: {issue['message']}{location}",
                flush=True,
            )

    return 1 if summary["error_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())