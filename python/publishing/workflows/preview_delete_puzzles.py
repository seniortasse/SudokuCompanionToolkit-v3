from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from python.publishing.cleanup import (
    DeleteAction,
    analyze_record_delete,
    decide_record_delete,
    render_delete_plan,
    render_policy_decision,
    render_report_summary,
    write_delete_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview archive/delete safety for canonical puzzle records."
    )
    parser.add_argument(
        "--record-id",
        required=True,
        action="append",
        dest="record_ids",
        help="Repeat --record-id to preview multiple records.",
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
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    requested_action = str(args.action)
    blocked_count = 0

    for index, record_id in enumerate(args.record_ids, start=1):
        plan = analyze_record_delete(
            record_id=record_id,
            library_id=args.library_id,
            requested_action=requested_action,
            records_dir=Path(args.records_dir),
            inventory_dir=Path(args.inventory_dir),
            books_dir=Path(args.books_dir),
            publications_dir=Path(args.publications_dir),
        )
        decision = decide_record_delete(plan)

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
                "workflow": "preview_delete_puzzles",
                "record_id": record_id,
            },
        )
        print("", flush=True)
        print(render_report_summary(report_path, stage="preview"), flush=True)

        if decision.outcome == "BLOCK":
            blocked_count += 1

    return 1 if blocked_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())