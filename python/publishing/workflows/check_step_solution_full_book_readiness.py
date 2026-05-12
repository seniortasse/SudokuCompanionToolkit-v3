from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from python.publishing.step_solutions.full_book_readiness import (
    DEFAULT_BOOKS_ROOT,
    DEFAULT_PACKAGES_ROOT,
    check_full_book_step_solution_readiness,
    write_full_book_readiness_report,
)
from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    normalize_step_solution_locale,
)


DEFAULT_READINESS_REPORT = (
    Path("runs/publishing/classic9/step_solution_full_book_readiness")
    / "full_book_readiness_report.json"
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
        description="Check full-book readiness for localized step-solution package production."
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
        help="Locales to check. Supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--books-root",
        type=Path,
        default=DEFAULT_BOOKS_ROOT,
        help="Root folder containing book folders.",
    )
    parser.add_argument(
        "--packages-root",
        type=Path,
        default=DEFAULT_PACKAGES_ROOT,
        help="Root folder for generated step-solution packages.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Root solution_templates folder.",
    )
    parser.add_argument(
        "--require-clean-output",
        action="store_true",
        help="Fail if existing output files are found.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_READINESS_REPORT,
        help="Output readiness report JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    payload = check_full_book_step_solution_readiness(
        book_id=args.book_id,
        locales=locales,
        books_root=args.books_root,
        packages_root=args.packages_root,
        templates_root=args.templates_root,
        require_clean_output=args.require_clean_output,
    )

    write_full_book_readiness_report(payload, args.report_json)
    payload["report_json"] = str(args.report_json)

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())