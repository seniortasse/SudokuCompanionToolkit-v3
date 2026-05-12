from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from python.publishing.step_solutions.csv_index_writer import DEFAULT_MAX_STEP_COLUMNS
from python.publishing.step_solutions.locale_templates import (
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.log_generator import DEFAULT_LEGACY_ROOT
from python.publishing.step_solutions.package_exporter import (
    export_step_solution_package,
)


def _split_values(values: List[str]) -> List[str]:
    """
    Support both:
        --locales en fr de
    and:
        --locales en,fr,de
    """

    out: List[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete localized step-solution package(s): "
            "Excel user logs, answer/step PNG images, sudokuIndexFile.csv, "
            "manifest, and package reports."
        )
    )

    book_group = parser.add_mutually_exclusive_group(required=True)
    book_group.add_argument(
        "--book-id",
        help="Single book id, for example BK-CL9-DW-B01.",
    )
    book_group.add_argument(
        "--book-ids",
        nargs="+",
        help="One or more book ids.",
    )

    locale_group = parser.add_mutually_exclusive_group(required=True)
    locale_group.add_argument(
        "--locale",
        help="Single locale, for example en, fr, de, it, es.",
    )
    locale_group.add_argument(
        "--locales",
        nargs="+",
        help="One or more locales. Supports space-separated or comma-separated values.",
    )

    parser.add_argument(
        "--books-root",
        type=Path,
        default=Path("datasets/sudoku_books/classic9/books"),
        help="Root folder containing book folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/sudoku_books/classic9/step_solution_packages"),
        help="Root folder for step-solution packages.",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help="Legacy python/step-by-step_solutions folder.",
    )
    parser.add_argument(
        "--legacy-command",
        default=None,
        help=(
            "Optional command template used to invoke the old user-log generator. "
            "Available placeholders: {input_json}, {output_xlsx}, {locale}, "
            "{visual_template}, {message_template}, {legacy_root}."
        ),
    )

    parser.add_argument(
        "--only-puzzle",
        default=None,
        help="Optional puzzle selector. Accepts L1-001 or L-1-1.",
    )
    parser.add_argument(
        "--only-section",
        default=None,
        help="Optional section selector. Accepts L1, L-1, or SEC-L1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of puzzles to process per book-locale package.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated logs/images/CSV outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing user logs during log generation.",
    )
    parser.add_argument(
        "--excel-visible",
        action="store_true",
        help="Show Excel while exporting images.",
    )
    parser.add_argument(
        "--max-step-columns",
        type=int,
        default=DEFAULT_MAX_STEP_COLUMNS,
        help="Number of Step/Explanation column pairs in sudokuIndexFile.csv.",
    )
    parser.add_argument(
        "--include-failed-csv-rows",
        action="store_true",
        help="Include failed image export rows in sudokuIndexFile.csv.",
    )

    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="Only generate Excel user_logs; skip images and CSV.",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only export images from existing user_logs; then skip CSV unless csv-only is run separately.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only regenerate sudokuIndexFile.csv from existing image_export_report.json.",
    )
    parser.add_argument(
        "--dry-run-logs",
        action="store_true",
        help="Create legacy input JSON files and manifest, but do not generate user logs/images/CSV.",
    )

    parser.add_argument(
        "--clipboard-retries",
        type=int,
        default=8,
        help="Clipboard retry attempts per copied Excel image.",
    )
    parser.add_argument(
        "--clipboard-sleep-seconds",
        type=float,
        default=0.12,
        help="Sleep duration between Excel CopyPicture and clipboard reads.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.logs_only and args.images_only:
        raise SystemExit("--logs-only and --images-only cannot be used together.")
    if args.logs_only and args.csv_only:
        raise SystemExit("--logs-only and --csv-only cannot be used together.")
    if args.images_only and args.csv_only:
        raise SystemExit("--images-only and --csv-only cannot be used together.")

    book_ids = [args.book_id] if args.book_id else _split_values(args.book_ids)
    locales = [args.locale] if args.locale else _split_values(args.locales)
    locales = [normalize_step_solution_locale(locale) for locale in locales]

    results = []

    for book_id in book_ids:
        for locale in locales:
            result = export_step_solution_package(
                book_id=book_id,
                locale=locale,
                books_root=args.books_root,
                output_root=args.output_root,
                legacy_root=args.legacy_root,
                legacy_command=args.legacy_command,
                only_puzzle=args.only_puzzle,
                only_section=args.only_section,
                limit=args.limit,
                force=args.force,
                skip_existing=args.skip_existing,
                excel_visible=args.excel_visible,
                max_step_columns=args.max_step_columns,
                include_failed_csv_rows=args.include_failed_csv_rows,
                logs_only=args.logs_only,
                images_only=args.images_only,
                csv_only=args.csv_only,
                dry_run_logs=args.dry_run_logs,
                clipboard_retries=args.clipboard_retries,
                clipboard_sleep_seconds=args.clipboard_sleep_seconds,
            )
            results.append(result)

    summary = {
        "phase": "phase_7_step_solution_package_export",
        "package_count": len(results),
        "ok": sum(1 for result in results if result.status == "ok"),
        "failed": sum(1 for result in results if result.status == "failed"),
        "dry_run": sum(1 for result in results if result.status == "dry_run"),
        "results": [result.to_dict() for result in results],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())