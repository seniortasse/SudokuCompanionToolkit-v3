from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Sequence

from python.publishing.cleanup import (
    DeleteAction,
    analyze_book_delete,
    confirmation_token_for_plan,
    decide_book_delete,
    is_confirmation_satisfied,
    prompt_for_confirmation,
    render_delete_plan,
    render_policy_decision,
    render_report_summary,
    render_snapshot_summary,
    restore_backup_snapshot,
    write_backup_snapshot,
    write_delete_report,
)
from python.publishing.inventory.assignment_ledger import unregister_assignments_for_book
from python.publishing.inventory.library_inventory_store import load_library_inventory, save_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index, save_catalog_index, update_record_status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete built books safely with dependency checks, preview, confirmation, and rollback."
    )
    parser.add_argument(
        "--book-id",
        required=True,
        action="append",
        dest="book_ids",
        help="Repeat --book-id to delete multiple books.",
    )
    parser.add_argument("--library-id", required=True)
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
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
    )
    parser.add_argument(
        "--records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
    )
    parser.add_argument(
        "--backup-root",
        default="runs/publishing/backups",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    parser.add_argument(
        "--cascade-publications",
        action="store_true",
        help="Also remove matching downstream publication directories.",
    )
    parser.add_argument(
        "--cascade-publication-specs",
        action="store_true",
        help="Also remove matching downstream publication spec files.",
    )
    parser.add_argument(
        "--delete-book-spec",
        action="store_true",
        help="Also remove matching book spec files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze, preview, and report without mutating anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass interactive confirmation.",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        help="Provide the required confirmation token explicitly.",
    )
    return parser.parse_args()


def _safe_remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _apply_book_delete_mutation(
    *,
    plan,
    decision,
    library_id: str,
    inventory_dir: Path,
    records_dir: Path,
    delete_book_spec: bool,
    cascade_publications: bool,
    cascade_publication_specs: bool,
) -> str:
    book_id = plan.target.target_id
    book_dir = Path(plan.metadata["book_dir"])

    inventory = load_library_inventory(
        base_dir=inventory_dir,
        library_id=library_id,
    )
    freed_record_ids = unregister_assignments_for_book(
        inventory,
        book_id=book_id,
    )
    save_library_inventory(
        inventory=inventory,
        base_dir=inventory_dir,
    )

    catalog_index = load_catalog_index(records_dir)
    for record_id in freed_record_ids:
        update_record_status(
            catalog_index,
            record_id=record_id,
            candidate_status="available",
        )
    save_catalog_index(catalog_index, records_dir)

    paths_to_remove: List[Path] = [book_dir]

    for dep in plan.dependencies:
        if dep.dependency_type == "book_spec" and delete_book_spec:
            paths_to_remove.append(Path(dep.path))
        elif dep.dependency_type == "publication_package" and cascade_publications:
            paths_to_remove.append(Path(dep.path))
        elif dep.dependency_type == "publication_spec" and cascade_publication_specs:
            paths_to_remove.append(Path(dep.path))

    seen = set()
    ordered_paths: List[Path] = []
    for item in paths_to_remove:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        ordered_paths.append(item)

    for path in ordered_paths:
        _safe_remove_path(path)

    removed_count = len([p for p in ordered_paths if not p.exists()])

    return (
        f"Deleted book {book_id}; freed {len(freed_record_ids)} assignment(s); "
        f"removed {removed_count} path(s)."
    )


def run_delete_books(
    *,
    book_ids: Sequence[str],
    library_id: str,
    books_dir: Path,
    book_specs_dir: Path,
    publications_dir: Path,
    publication_specs_dir: Path,
    inventory_dir: Path,
    records_dir: Path,
    backup_root: Path,
    report_dir: Path,
    cascade_publications: bool = False,
    cascade_publication_specs: bool = False,
    delete_book_spec: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    confirm: str | None = None,
) -> int:
    plans = []
    decisions = []
    blocked_count = 0

    for index, book_id in enumerate(book_ids, start=1):
        plan = analyze_book_delete(
            book_id=book_id,
            library_id=library_id,
            requested_action=DeleteAction.DELETE,
            books_dir=books_dir,
            book_specs_dir=book_specs_dir,
            publications_dir=publications_dir,
            publication_specs_dir=publication_specs_dir,
            inventory_dir=inventory_dir,
            records_dir=records_dir,
        )

        if delete_book_spec:
            for dep in plan.dependencies:
                if dep.dependency_type == "book_spec" and dep.path not in plan.files_to_delete:
                    plan.files_to_delete.append(dep.path)

        if cascade_publications:
            for dep in plan.dependencies:
                if dep.dependency_type == "publication_package" and dep.path not in plan.files_to_delete:
                    plan.files_to_delete.append(dep.path)

        if cascade_publication_specs:
            for dep in plan.dependencies:
                if dep.dependency_type == "publication_spec" and dep.path not in plan.files_to_delete:
                    plan.files_to_delete.append(dep.path)

        decision = decide_book_delete(
            plan,
            cascade_publications=bool(cascade_publications),
            cascade_publication_specs=bool(cascade_publication_specs),
        )

        plans.append(plan)
        decisions.append(decision)

        if index > 1:
            print("", flush=True)

        print(render_delete_plan(plan), flush=True)
        print("", flush=True)
        print(render_policy_decision(decision), flush=True)

        report_path = write_delete_report(
            plan=plan,
            decision=decision,
            report_dir=report_dir,
            stage="preview",
            extra={
                "workflow": "delete_books",
                "book_id": book_id,
                "cascade_publications": bool(cascade_publications),
                "cascade_publication_specs": bool(cascade_publication_specs),
                "delete_book_spec": bool(delete_book_spec),
            },
        )
        print("", flush=True)
        print(render_report_summary(report_path, stage="preview"), flush=True)

        if decision.outcome == "BLOCK":
            blocked_count += 1

    if blocked_count > 0:
        print("", flush=True)
        print(f"Blocked books: {blocked_count}. No mutation was performed.", flush=True)
        return 1

    if dry_run:
        print("", flush=True)
        print("Dry run requested. No mutation was performed.", flush=True)
        return 0

    first_plan = plans[0]
    is_batch = len(plans) > 1

    confirmed = is_confirmation_satisfied(
        plan=first_plan,
        provided_token=confirm,
        yes=yes,
        is_batch=is_batch,
        is_delete_all=False,
    )
    if not confirmed:
        spec = confirmation_token_for_plan(
            first_plan,
            is_batch=is_batch,
            is_delete_all=False,
        )
        print("", flush=True)
        print(
            f"Confirmation required before mutation. Expected token: {spec.required_token}",
            flush=True,
        )
        confirmed = prompt_for_confirmation(
            plan=first_plan,
            is_batch=is_batch,
            is_delete_all=False,
        )

    if not confirmed:
        print("Confirmation failed. No mutation was performed.", flush=True)
        return 1

    for plan, decision in zip(plans, decisions):
        snapshot_dir = write_backup_snapshot(
            plan=plan,
            backup_root=backup_root,
        )
        print("", flush=True)
        print(render_snapshot_summary(snapshot_dir), flush=True)

        try:
            message = _apply_book_delete_mutation(
                plan=plan,
                decision=decision,
                library_id=library_id,
                inventory_dir=inventory_dir,
                records_dir=records_dir,
                delete_book_spec=bool(delete_book_spec),
                cascade_publications=bool(cascade_publications),
                cascade_publication_specs=bool(cascade_publication_specs),
            )

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="execute",
                snapshot_dir=snapshot_dir,
                extra={
                    "workflow": "delete_books",
                    "message": message,
                    "cascade_publications": bool(cascade_publications),
                    "cascade_publication_specs": bool(cascade_publication_specs),
                    "delete_book_spec": bool(delete_book_spec),
                },
            )
            print(message, flush=True)
            print(render_report_summary(report_path, stage="execute"), flush=True)

        except Exception as exc:
            restored_paths = restore_backup_snapshot(snapshot_dir=snapshot_dir)
            rollback_report = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="rollback",
                snapshot_dir=snapshot_dir,
                extra={
                    "workflow": "delete_books",
                    "error": str(exc),
                    "restored_paths": restored_paths,
                },
            )
            print(f"Mutation failed for {plan.target.target_id}: {exc}", flush=True)
            print(render_report_summary(rollback_report, stage="rollback"), flush=True)
            return 1

    return 0


def main() -> int:
    args = _parse_args()
    return run_delete_books(
        book_ids=args.book_ids,
        library_id=args.library_id,
        books_dir=Path(args.books_dir),
        book_specs_dir=Path(args.book_specs_dir),
        publications_dir=Path(args.publications_dir),
        publication_specs_dir=Path(args.publication_specs_dir),
        inventory_dir=Path(args.inventory_dir),
        records_dir=Path(args.records_dir),
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        cascade_publications=bool(args.cascade_publications),
        cascade_publication_specs=bool(args.cascade_publication_specs),
        delete_book_spec=bool(args.delete_book_spec),
        dry_run=bool(args.dry_run),
        yes=bool(args.yes),
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())