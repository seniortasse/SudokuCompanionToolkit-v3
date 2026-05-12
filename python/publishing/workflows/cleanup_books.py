from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from python.publishing.cleanup import (
    DeleteAction,
    analyze_book_delete,
    confirmation_token_for_plan,
    is_confirmation_satisfied,
    prompt_for_confirmation,
)
from python.publishing.workflows.delete_books import run_delete_books


def _iter_book_dirs(books_dir: Path) -> List[Path]:
    if not books_dir.exists():
        return []
    return sorted(p for p in books_dir.iterdir() if p.is_dir())


def _select_book_ids(
    *,
    books_dir: Path,
    book_ids: List[str] | None,
    all_books: bool,
    prefix: str | None,
) -> List[str]:
    if book_ids:
        return sorted(dict.fromkeys(book_ids))

    discovered = [p.name for p in _iter_book_dirs(books_dir)]

    if prefix:
        discovered = [name for name in discovered if name.startswith(prefix)]

    if all_books:
        return discovered

    return []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk cleanup wrapper for built books."
    )
    parser.add_argument(
        "--book-id",
        action="append",
        dest="book_ids",
        help="Repeat --book-id to target specific books.",
    )
    parser.add_argument(
        "--all-books",
        action="store_true",
        help="Target all discovered built book directories.",
    )
    parser.add_argument(
        "--book-id-prefix",
        default=None,
        help="Optional prefix filter for discovered books, e.g. BK-CL9-DW-",
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
    )
    parser.add_argument(
        "--cascade-publication-specs",
        action="store_true",
    )
    parser.add_argument(
        "--delete-book-spec",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
    )
    parser.add_argument(
        "--confirm",
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    books_dir = Path(args.books_dir)
    selected_book_ids = _select_book_ids(
        books_dir=books_dir,
        book_ids=args.book_ids,
        all_books=bool(args.all_books),
        prefix=args.book_id_prefix,
    )

    if not selected_book_ids:
        print("No books matched the requested selection.", flush=True)
        return 1

    is_delete_all = bool(args.all_books and not args.book_ids and not args.book_id_prefix)
    is_batch = len(selected_book_ids) > 1

    if args.dry_run:
        return run_delete_books(
            book_ids=selected_book_ids,
            library_id=args.library_id,
            books_dir=books_dir,
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
            dry_run=True,
            yes=bool(args.yes),
            confirm=args.confirm,
        )

    first_plan = analyze_book_delete(
        book_id=selected_book_ids[0],
        library_id=args.library_id,
        requested_action=DeleteAction.DELETE,
        books_dir=books_dir,
        book_specs_dir=Path(args.book_specs_dir),
        publications_dir=Path(args.publications_dir),
        publication_specs_dir=Path(args.publication_specs_dir),
        inventory_dir=Path(args.inventory_dir),
        records_dir=Path(args.records_dir),
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
        print(f"Selected books: {len(selected_book_ids)}", flush=True)
        print(f"Expected confirmation token: {spec.required_token}", flush=True)
        confirmed = prompt_for_confirmation(
            plan=first_plan,
            is_batch=is_batch,
            is_delete_all=is_delete_all,
        )

    if not confirmed:
        print("Confirmation failed. No mutation was performed.", flush=True)
        return 1

    return run_delete_books(
        book_ids=selected_book_ids,
        library_id=args.library_id,
        books_dir=books_dir,
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
        dry_run=False,
        yes=True,
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())