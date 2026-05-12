from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.workflows.audit_integrity import audit_integrity
from python.publishing.workflows.cleanup_books import main as cleanup_books_main
from python.publishing.workflows.cleanup_generation_pool import main as cleanup_generation_pool_main
from python.publishing.workflows.cleanup_puzzle_catalog import main as cleanup_puzzle_catalog_main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-level cleanup entrypoint for the publishing platform."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "audit",
            "books",
            "puzzle_catalog",
            "generation_pool",
        ],
        help="Choose which cleanup subsystem to run.",
    )
    parser.add_argument(
        "--library-id",
        default=None,
        help="Required for audit, books, and puzzle_catalog modes.",
    )
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
    args, unknown = parser.parse_known_args()
    args.unknown = unknown
    return args


def _run_audit(args: argparse.Namespace) -> int:
    if not args.library_id:
        print("--library-id is required for audit mode.", flush=True)
        return 1

    report = audit_integrity(
        library_id=args.library_id,
        records_dir=Path(args.records_dir),
        inventory_dir=Path(args.inventory_dir),
        books_dir=Path(args.books_dir),
        book_specs_dir=Path(args.book_specs_dir),
        publications_dir=Path(args.publications_dir),
        publication_specs_dir=Path(args.publication_specs_dir),
    )

    summary = report["summary"]
    print("=" * 72, flush=True)
    print("PLATFORM AUDIT", flush=True)
    print("=" * 72, flush=True)
    print(f"Library id:       {args.library_id}", flush=True)
    print(f"Errors:           {summary['error_count']}", flush=True)
    print(f"Warnings:         {summary['warning_count']}", flush=True)
    print(f"Total issues:     {summary['issue_count']}", flush=True)
    return 1 if summary["error_count"] > 0 else 0


def main() -> int:
    args = _parse_args()

    if args.mode == "audit":
        return _run_audit(args)

    if args.mode == "books":
        return cleanup_books_main()

    if args.mode == "puzzle_catalog":
        return cleanup_puzzle_catalog_main()

    if args.mode == "generation_pool":
        return cleanup_generation_pool_main()

    print(f"Unsupported mode: {args.mode}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())