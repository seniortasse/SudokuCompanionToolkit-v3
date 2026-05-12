from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.cleanup import (
    DeleteAction,
    analyze_book_delete,
    decide_book_delete,
    render_delete_plan,
    render_policy_decision,
    render_report_summary,
    write_delete_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview delete safety for built books."
    )
    parser.add_argument(
        "--book-id",
        required=True,
        action="append",
        dest="book_ids",
        help="Repeat --book-id to preview multiple books.",
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
        "--cascade-publications",
        action="store_true",
        help="Preview as if downstream publication directories would also be removed.",
    )
    parser.add_argument(
        "--cascade-publication-specs",
        action="store_true",
        help="Preview as if downstream publication spec files would also be removed.",
    )
    parser.add_argument(
        "--delete-book-spec",
        action="store_true",
        help="Preview as if matching book spec files would also be removed.",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    blocked_count = 0

    for index, book_id in enumerate(args.book_ids, start=1):
        plan = analyze_book_delete(
            book_id=book_id,
            library_id=args.library_id,
            requested_action=DeleteAction.DELETE,
            books_dir=Path(args.books_dir),
            book_specs_dir=Path(args.book_specs_dir),
            publications_dir=Path(args.publications_dir),
            publication_specs_dir=Path(args.publication_specs_dir),
            inventory_dir=Path(args.inventory_dir),
            records_dir=Path(args.records_dir),
        )

        if args.delete_book_spec:
            for dep in plan.dependencies:
                if dep.dependency_type == "book_spec" and dep.path not in plan.files_to_delete:
                    plan.files_to_delete.append(dep.path)

        if args.cascade_publications:
            for dep in plan.dependencies:
                if dep.dependency_type == "publication_package" and dep.path not in plan.files_to_delete:
                    plan.files_to_delete.append(dep.path)

        if args.cascade_publication_specs:
            for dep in plan.dependencies:
                if dep.dependency_type == "publication_spec" and dep.path not in plan.files_to_delete:
                    plan.files_to_delete.append(dep.path)

        decision = decide_book_delete(
            plan,
            cascade_publications=bool(args.cascade_publications),
            cascade_publication_specs=bool(args.cascade_publication_specs),
        )

        if index > 1:
            print("", flush=True)

        print(render_delete_plan(plan), flush=True)
        print("", flush=True)
        print(render_policy_decision(decision), flush=True)

        report_path = write_delete_report(
            plan=plan,
            decision=decision,
            report_dir=Path(args.report_dir),
            stage="preview",
            extra={
                "workflow": "preview_delete_book",
                "book_id": book_id,
                "cascade_publications": bool(args.cascade_publications),
                "cascade_publication_specs": bool(args.cascade_publication_specs),
                "delete_book_spec": bool(args.delete_book_spec),
            },
        )
        print("", flush=True)
        print(render_report_summary(report_path, stage="preview"), flush=True)

        if decision.outcome == "BLOCK":
            blocked_count += 1

    return 1 if blocked_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())