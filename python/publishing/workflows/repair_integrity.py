from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from python.publishing.cleanup import (
    render_report_summary,
    render_snapshot_summary,
    write_backup_snapshot,
    write_delete_report,
)
from python.publishing.cleanup.delete_models import DeleteAction, DeletePlan, DeleteTarget
# Keep repair logic self-contained to avoid depending on helper names/signatures
# that may vary across assignment_ledger revisions.
from python.publishing.inventory.library_inventory_store import load_library_inventory, save_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index, save_catalog_index, update_record_status
from python.publishing.workflows.audit_integrity import audit_integrity


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _build_repair_plan(
    *,
    library_id: str,
    inventory_dir: Path,
    records_dir: Path,
) -> DeletePlan:
    return DeletePlan(
        target=DeleteTarget(
            target_type="integrity_repair",
            target_id=library_id,
            display_name=library_id,
        ),
        requested_action=DeleteAction.CASCADE_DELETE,
        allowed_actions=[DeleteAction.CASCADE_DELETE],
        files_to_update=[
            _normalize_path(inventory_dir / "_library_inventory.json"),
            _normalize_path(records_dir / "_catalog_index.json"),
        ],
        files_to_delete=[],
        notes=[
            "Integrity repair may update inventory assignment entries and catalog statuses.",
            "Repair is conservative and only applies safe, deterministic fixes.",
        ],
        metadata={
            "library_id": library_id,
        },
    )


def _safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_book_dirs(books_dir: Path) -> List[Path]:
    if not books_dir.exists():
        return []
    return sorted(p for p in books_dir.iterdir() if p.is_dir())


def _remove_assignments_for_missing_books_in_place(
    inventory: Dict[str, Any],
    *,
    record_id: str,
    existing_book_ids: set[str],
) -> None:
    records = inventory.setdefault("records", {})
    entry = records.get(record_id)
    if not entry:
        return

    assignments = list(entry.get("assignments", []))
    kept = [
        item for item in assignments
        if str(item.get("book_id", "")).strip() in existing_book_ids
    ]
    entry["assignments"] = kept
    entry["assignment_count"] = len(kept)


def _add_assignment_in_place(
    inventory: Dict[str, Any],
    *,
    record_id: str,
    book_id: str,
) -> None:
    records = inventory.setdefault("records", {})
    entry = records.setdefault(record_id, {})

    assignments = list(entry.get("assignments", []))
    if any(str(item.get("book_id", "")).strip() == book_id for item in assignments):
        entry["assignment_count"] = len(assignments)
        return

    assignments.append({"book_id": book_id})
    entry["assignments"] = assignments
    entry["assignment_count"] = len(assignments)


def _rebuild_book_assignment_map(books_dir: Path) -> Dict[str, List[str]]:
    from python.publishing.book_builder.book_package_store import load_built_book_package

    result: Dict[str, List[str]] = {}
    for book_dir in _iter_book_dirs(books_dir):
        try:
            _manifest, _sections, assigned_puzzles = load_built_book_package(book_dir)
        except Exception:
            continue

        record_ids: List[str] = []
        for assigned in assigned_puzzles:
            record_id = str(getattr(assigned, "record_id", "")).strip()
            if record_id:
                record_ids.append(record_id)
        result[book_dir.name] = sorted(set(record_ids))
    return result


def repair_integrity(
    *,
    library_id: str,
    records_dir: Path,
    inventory_dir: Path,
    books_dir: Path,
    book_specs_dir: Path,
    publications_dir: Path,
    publication_specs_dir: Path,
) -> Dict[str, Any]:
    inventory = load_library_inventory(base_dir=inventory_dir, library_id=library_id)
    catalog_index = load_catalog_index(records_dir)

    inventory_records = dict(inventory.get("records", {}))
    index_records_by_id = dict(catalog_index.get("records_by_id", {}))
    book_assignment_map = _rebuild_book_assignment_map(books_dir)

    repaired_actions: List[Dict[str, Any]] = []
    skipped_actions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1) Remove inventory assignments pointing to missing books
    # ------------------------------------------------------------------
    for record_id in sorted(list(inventory_records.keys())):
        entry = inventory_records.get(record_id, {})
        assignments = list(entry.get("assignments", []))
        before_count = len(assignments)

        _remove_assignments_for_missing_books_in_place(
            inventory,
            record_id=record_id,
            existing_book_ids=set(book_assignment_map.keys()),
        )

        after_entry = inventory.get("records", {}).get(record_id, {})
        after_assignments = list(after_entry.get("assignments", []))
        after_count = len(after_assignments)

        if after_count != before_count:
            repaired_actions.append({
                "action": "remove_assignments_for_missing_book",
                "record_id": record_id,
                "before_count": before_count,
                "after_count": after_count,
            })

    # ------------------------------------------------------------------
    # 2) Re-add missing inventory assignments for records that are present
    #    in built books but missing from inventory assignments.
    # ------------------------------------------------------------------
    inventory = load_library_inventory(base_dir=inventory_dir, library_id=library_id)
    for book_id, record_ids in sorted(book_assignment_map.items()):
        for record_id in record_ids:
            entry = dict(inventory.get("records", {}).get(record_id, {}))
            assignments = list(entry.get("assignments", []))

            if any(str(a.get("book_id", "")).strip() == book_id for a in assignments):
                continue

            if record_id not in index_records_by_id:
                skipped_actions.append({
                    "action": "recreate_assignment_missing_index_entry",
                    "record_id": record_id,
                    "book_id": book_id,
                    "reason": "record missing from _catalog_index.json",
                })
                continue

            try:
                _add_assignment_in_place(
                    inventory,
                    record_id=record_id,
                    book_id=book_id,
                )
                repaired_actions.append({
                    "action": "recreate_inventory_assignment_from_built_book",
                    "record_id": record_id,
                    "book_id": book_id,
                })
            except Exception as exc:
                skipped_actions.append({
                    "action": "recreate_inventory_assignment_from_built_book",
                    "record_id": record_id,
                    "book_id": book_id,
                    "reason": str(exc),
                })

    save_library_inventory(
        inventory=inventory,
        base_dir=inventory_dir,
    )

    # ------------------------------------------------------------------
    # 3) Sync candidate_status between inventory reality and catalog index.
    #    If a record has one or more assignments -> assigned
    #    Else -> available, unless index already says archived
    # ------------------------------------------------------------------
    inventory = load_library_inventory(base_dir=inventory_dir, library_id=library_id)
    for record_id, index_entry in sorted(index_records_by_id.items()):
        inventory_entry = dict(inventory.get("records", {}).get(record_id, {}))
        assignments = list(inventory_entry.get("assignments", []))
        current_status = str(index_entry.get("candidate_status", "")).strip().lower()

        if assignments:
            desired_status = "assigned"
        else:
            desired_status = "archived" if current_status == "archived" else "available"

        if current_status != desired_status:
            update_record_status(
                catalog_index,
                record_id=record_id,
                candidate_status=desired_status,
            )
            repaired_actions.append({
                "action": "sync_catalog_index_candidate_status",
                "record_id": record_id,
                "from_status": current_status,
                "to_status": desired_status,
            })

    # ------------------------------------------------------------------
    # 4) Sync inventory candidate_status to match assignment reality too.
    # ------------------------------------------------------------------
    for record_id, entry in sorted(inventory.get("records", {}).items()):
        assignments = list(entry.get("assignments", []))
        current_status = str(entry.get("candidate_status", "")).strip().lower()
        desired_status = "assigned" if assignments else "available"

        if current_status != desired_status:
            entry["candidate_status"] = desired_status
            repaired_actions.append({
                "action": "sync_inventory_candidate_status",
                "record_id": record_id,
                "from_status": current_status,
                "to_status": desired_status,
            })

        entry["assignment_count"] = len(assignments)

    save_library_inventory(
        inventory=inventory,
        base_dir=inventory_dir,
    )
    save_catalog_index(catalog_index, records_dir)

    post_audit = audit_integrity(
        library_id=library_id,
        records_dir=records_dir,
        inventory_dir=inventory_dir,
        books_dir=books_dir,
        book_specs_dir=book_specs_dir,
        publications_dir=publications_dir,
        publication_specs_dir=publication_specs_dir,
    )

    return {
        "generated_utc": _utc_stamp(),
        "library_id": library_id,
        "repaired_actions": repaired_actions,
        "skipped_actions": skipped_actions,
        "post_repair_audit": post_audit,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair safe and deterministic integrity issues in inventory/catalog state."
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
        "--backup-root",
        default="runs/publishing/backups",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/integrity_reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pre-repair audit and write a preview report only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    records_dir = Path(args.records_dir)
    inventory_dir = Path(args.inventory_dir)
    books_dir = Path(args.books_dir)
    book_specs_dir = Path(args.book_specs_dir)
    publications_dir = Path(args.publications_dir)
    publication_specs_dir = Path(args.publication_specs_dir)
    backup_root = Path(args.backup_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    pre_audit = audit_integrity(
        library_id=args.library_id,
        records_dir=records_dir,
        inventory_dir=inventory_dir,
        books_dir=books_dir,
        book_specs_dir=book_specs_dir,
        publications_dir=publications_dir,
        publication_specs_dir=publication_specs_dir,
    )

    repair_plan = _build_repair_plan(
        library_id=args.library_id,
        inventory_dir=inventory_dir,
        records_dir=records_dir,
    )

    preview_report = write_delete_report(
        plan=repair_plan,
        decision=None,
        report_dir=report_dir,
        stage="preview",
        extra={
            "workflow": "repair_integrity",
            "pre_audit": pre_audit,
            "dry_run": bool(args.dry_run),
        },
    )
    print(render_report_summary(preview_report, stage="preview"), flush=True)

    if args.dry_run:
        summary = pre_audit["summary"]
        print("=" * 72, flush=True)
        print("INTEGRITY REPAIR PREVIEW", flush=True)
        print("=" * 72, flush=True)
        print(f"Library id:   {args.library_id}", flush=True)
        print(f"Errors:       {summary['error_count']}", flush=True)
        print(f"Warnings:     {summary['warning_count']}", flush=True)
        print(f"Report path:  {preview_report}", flush=True)
        print("Dry run requested. No mutation was performed.", flush=True)
        return 0

    snapshot_dir = write_backup_snapshot(
        plan=repair_plan,
        backup_root=backup_root,
    )
    print(render_snapshot_summary(snapshot_dir), flush=True)

    try:
        repair_report = repair_integrity(
            library_id=args.library_id,
            records_dir=records_dir,
            inventory_dir=inventory_dir,
            books_dir=books_dir,
            book_specs_dir=book_specs_dir,
            publications_dir=publications_dir,
            publication_specs_dir=publication_specs_dir,
        )

        execute_report = write_delete_report(
            plan=repair_plan,
            decision=None,
            report_dir=report_dir,
            stage="execute",
            snapshot_dir=snapshot_dir,
            extra={
                "workflow": "repair_integrity",
                "repair_report": repair_report,
            },
        )
        print(render_report_summary(execute_report, stage="execute"), flush=True)

        post_summary = repair_report["post_repair_audit"]["summary"]
        print("=" * 72, flush=True)
        print("INTEGRITY REPAIR COMPLETE", flush=True)
        print("=" * 72, flush=True)
        print(f"Repaired actions: {len(repair_report['repaired_actions'])}", flush=True)
        print(f"Skipped actions:  {len(repair_report['skipped_actions'])}", flush=True)
        print(f"Remaining errors: {post_summary['error_count']}", flush=True)
        print(f"Remaining warns:  {post_summary['warning_count']}", flush=True)
        return 0 if post_summary["error_count"] == 0 else 1

    except Exception as exc:
        from python.publishing.cleanup import restore_backup_snapshot

        restored_paths = restore_backup_snapshot(snapshot_dir=snapshot_dir)
        rollback_report = write_delete_report(
            plan=repair_plan,
            decision=None,
            report_dir=report_dir,
            stage="rollback",
            snapshot_dir=snapshot_dir,
            extra={
                "workflow": "repair_integrity",
                "error": str(exc),
                "restored_paths": restored_paths,
            },
        )
        print(f"Repair failed: {exc}", flush=True)
        print(render_report_summary(rollback_report, stage="rollback"), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())