from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.workflows.delete_books import run_delete_books


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete a built book safely. Legacy compatibility wrapper."
    )
    parser.add_argument("--book-id", required=True, action="append", dest="book_ids")
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