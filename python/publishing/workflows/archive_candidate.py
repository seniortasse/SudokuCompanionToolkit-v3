from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.cleanup.delete_models import DeleteAction
from python.publishing.workflows.delete_puzzles import run_delete_puzzles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive canonical puzzle records safely. Legacy compatibility wrapper."
    )
    parser.add_argument("--record-id", required=True, action="append", dest="record_ids")
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
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--confirm", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_delete_puzzles(
        record_ids=args.record_ids,
        action=DeleteAction.ARCHIVE,
        library_id=args.library_id,
        records_dir=Path(args.records_dir),
        inventory_dir=Path(args.inventory_dir),
        books_dir=Path(args.books_dir),
        publications_dir=Path(args.publications_dir),
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        dry_run=bool(args.dry_run),
        yes=bool(args.yes),
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())