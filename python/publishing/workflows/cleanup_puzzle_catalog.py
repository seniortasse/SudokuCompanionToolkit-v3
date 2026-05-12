from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from python.publishing.cleanup import (
    DeleteAction,
    analyze_record_delete,
    confirmation_token_for_plan,
    is_confirmation_satisfied,
    prompt_for_confirmation,
)
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index
from python.publishing.workflows.delete_puzzles import run_delete_puzzles


def _select_record_ids(
    *,
    records_dir: Path,
    record_ids: List[str] | None,
    library_id: str,
    inventory_dir: Path,
    all_available: bool,
    all_archived: bool,
    all_orphaned: bool,
    books_dir: Path,
    publications_dir: Path,
) -> List[str]:
    if record_ids:
        return sorted(dict.fromkeys(record_ids))

    catalog_index = load_catalog_index(records_dir)
    records_by_id = dict(catalog_index.get("records_by_id", {}))
    selected: List[str] = []

    for record_id, entry in sorted(records_by_id.items()):
        status = str(entry.get("candidate_status", "")).strip().lower()

        if all_available and status == "available":
            selected.append(record_id)
            continue

        if all_archived and status == "archived":
            selected.append(record_id)
            continue

        if all_orphaned:
            plan = analyze_record_delete(
                record_id=record_id,
                library_id=library_id,
                requested_action=DeleteAction.DELETE,
                records_dir=records_dir,
                inventory_dir=inventory_dir,
                books_dir=books_dir,
                publications_dir=publications_dir,
            )
            if not plan.dependencies:
                selected.append(record_id)

    return sorted(dict.fromkeys(selected))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk cleanup wrapper for canonical puzzle records."
    )
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help="Repeat --record-id to target specific canonical records.",
    )
    parser.add_argument(
        "--all-available",
        action="store_true",
        help="Target all canonical records whose catalog status is available.",
    )
    parser.add_argument(
        "--all-archived",
        action="store_true",
        help="Target all canonical records whose catalog status is archived.",
    )
    parser.add_argument(
        "--all-orphaned",
        action="store_true",
        help="Target records with no detected downstream dependencies.",
    )
    parser.add_argument(
        "--action",
        choices=[DeleteAction.ARCHIVE, DeleteAction.DELETE],
        default=DeleteAction.DELETE,
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
        "--publications-dir",
        default="datasets/sudoku_books/classic9/publications",
    )
    parser.add_argument(
        "--backup-root",
        default="runs/publishing/backups",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--confirm", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    records_dir = Path(args.records_dir)
    inventory_dir = Path(args.inventory_dir)
    books_dir = Path(args.books_dir)
    publications_dir = Path(args.publications_dir)

    selected_record_ids = _select_record_ids(
        records_dir=records_dir,
        record_ids=args.record_ids,
        library_id=args.library_id,
        inventory_dir=inventory_dir,
        all_available=bool(args.all_available),
        all_archived=bool(args.all_archived),
        all_orphaned=bool(args.all_orphaned),
        books_dir=books_dir,
        publications_dir=publications_dir,
    )

    if not selected_record_ids:
        print("No canonical puzzle records matched the requested selection.", flush=True)
        return 1

    is_delete_all = bool(
        (args.all_available or args.all_archived or args.all_orphaned) and not args.record_ids
    )
    is_batch = len(selected_record_ids) > 1

    if args.dry_run:
        return run_delete_puzzles(
            record_ids=selected_record_ids,
            action=str(args.action),
            library_id=args.library_id,
            records_dir=records_dir,
            inventory_dir=inventory_dir,
            books_dir=books_dir,
            publications_dir=publications_dir,
            backup_root=Path(args.backup_root),
            report_dir=Path(args.report_dir),
            dry_run=True,
            yes=bool(args.yes),
            confirm=args.confirm,
        )

    first_plan = analyze_record_delete(
        record_id=selected_record_ids[0],
        library_id=args.library_id,
        requested_action=str(args.action),
        records_dir=records_dir,
        inventory_dir=inventory_dir,
        books_dir=books_dir,
        publications_dir=publications_dir,
    )

    confirmed = is_confirmation_satisfied(
        plan=first_plan,
        provided_token=args.confirm,
        yes=bool(args.yes),
        is_batch=is_batch,
        is_delete_all=is_delete_all,
    )
    if not confirmed:
        spec = confirmation_token_for_plan(
            first_plan,
            is_batch=is_batch,
            is_delete_all=is_delete_all,
        )
        print(f"Selected canonical records: {len(selected_record_ids)}", flush=True)
        print(f"Expected confirmation token: {spec.required_token}", flush=True)
        confirmed = prompt_for_confirmation(
            plan=first_plan,
            is_batch=is_batch,
            is_delete_all=is_delete_all,
        )

    if not confirmed:
        print("Confirmation failed. No mutation was performed.", flush=True)
        return 1

    return run_delete_puzzles(
        record_ids=selected_record_ids,
        action=str(args.action),
        library_id=args.library_id,
        records_dir=records_dir,
        inventory_dir=inventory_dir,
        books_dir=books_dir,
        publications_dir=publications_dir,
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        dry_run=False,
        yes=True,
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())