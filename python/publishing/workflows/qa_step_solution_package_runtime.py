from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.runtime_qa import (
    DEFAULT_BOOKS_ROOT,
    DEFAULT_STEP_SOLUTION_PACKAGES_ROOT,
    qa_multiple_step_solution_runtime_packages,
    write_runtime_qa_report,
)


def _split_values(values: List[str]) -> List[str]:
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
            "QA generated step-solution package runtime outputs: user logs, "
            "PNG images, sudokuIndexFile.csv, reports, and localized narratives."
        )
    )
    parser.add_argument(
        "--book-id",
        required=True,
        help="Book id, for example BK-CL9-DW-B01.",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        required=True,
        help="Locales to inspect. Supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_STEP_SOLUTION_PACKAGES_ROOT,
        help="Root folder for step-solution packages.",
    )
    parser.add_argument(
        "--books-root",
        type=Path,
        default=DEFAULT_BOOKS_ROOT,
        help="Root folder containing book folders.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Root solution_templates folder.",
    )
    parser.add_argument(
        "--max-workbooks-to-check",
        type=int,
        default=3,
        help="Maximum generated user-log workbooks to inspect per locale.",
    )
    parser.add_argument(
        "--max-csv-rows-to-check",
        type=int,
        default=25,
        help="Maximum sudokuIndexFile.csv rows to inspect per locale.",
    )
    parser.add_argument(
        "--allow-english-leakage",
        action="store_true",
        help="Downgrade detected English leakage from error to warning.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help=(
            "Optional explicit QA report path. Defaults to "
            "<output-root>/<book-id>-<first-locale>/reports/runtime_qa_report.json "
            "for one locale, or <output-root>/<book-id>__multi_locale_runtime_qa_report.json "
            "for multiple locales."
        ),
    )
    return parser.parse_args()


def _default_report_path(
    output_root: Path,
    book_id: str,
    locales: List[str],
) -> Path:
    if len(locales) == 1:
        return (
            Path(output_root)
            / f"{book_id}-{locales[0]}"
            / "reports"
            / "runtime_qa_report.json"
        )

    return Path(output_root) / f"{book_id}__multi_locale_runtime_qa_report.json"


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    payload = qa_multiple_step_solution_runtime_packages(
        book_id=args.book_id,
        locales=locales,
        output_root=args.output_root,
        books_root=args.books_root,
        templates_root=args.templates_root,
        max_workbooks_to_check=args.max_workbooks_to_check,
        max_csv_rows_to_check=args.max_csv_rows_to_check,
        require_localized_text=not args.allow_english_leakage,
    )

    report_path = args.report_json or _default_report_path(
        output_root=args.output_root,
        book_id=args.book_id,
        locales=locales,
    )
    write_runtime_qa_report(payload, report_path)

    payload["runtime_qa_report"] = str(report_path)

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())