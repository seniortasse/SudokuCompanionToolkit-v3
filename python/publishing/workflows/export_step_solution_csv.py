from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.step_solutions.csv_index_writer import (
    DEFAULT_MAX_STEP_COLUMNS,
    build_csv_summary,
    write_sudoku_index_csv_from_image_report,
)
from python.publishing.step_solutions.locale_templates import normalize_step_solution_locale
from python.publishing.step_solutions.models import StepSolutionPackageRequest
from python.publishing.step_solutions.package_manifest import write_json
from python.publishing.step_solutions.paths import (
    ensure_package_directories,
    resolve_package_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate external-compatible sudokuIndexFile.csv from Phase 5 "
            "image_export_report.json."
        )
    )
    parser.add_argument(
        "--book-id",
        required=True,
        help="Book id, for example BK-CL9-DW-B01.",
    )
    parser.add_argument(
        "--locale",
        default="en",
        help="Step-solution locale, for example en, fr, de, it, es.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/sudoku_books/classic9/step_solution_packages"),
        help="Root folder for step-solution packages.",
    )
    parser.add_argument(
        "--image-export-report",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to image_export_report.json. "
            "Defaults to package reports/image_export_report.json."
        ),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit CSV output path. "
            "Defaults to package sudokuIndexFile.csv."
        ),
    )
    parser.add_argument(
        "--max-step-columns",
        type=int,
        default=DEFAULT_MAX_STEP_COLUMNS,
        help="Number of Step/Explanation column pairs to write. Default: 40.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed Phase 5 image-export results as mostly blank rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locale = normalize_step_solution_locale(args.locale)

    request = StepSolutionPackageRequest(
        book_id=args.book_id,
        locale=locale,
        output_root=args.output_root,
    )

    package_paths = resolve_package_paths(request)
    ensure_package_directories(package_paths)

    report_path = args.image_export_report or (
        package_paths.reports_dir / "image_export_report.json"
    )
    csv_path = args.csv_path or package_paths.sudoku_index_csv_path

    written_csv_path = write_sudoku_index_csv_from_image_report(
        report_path=report_path,
        csv_path=csv_path,
        package_paths=package_paths,
        max_step_columns=args.max_step_columns,
        include_failed=args.include_failed,
    )

    summary = build_csv_summary(written_csv_path)

    report = {
        "phase": "phase_6_sudoku_index_csv",
        "book_id": request.book_id,
        "locale": locale,
        "package_id": request.package_id(),
        "package_root": str(package_paths.package_root),
        "image_export_report": str(report_path),
        "sudoku_index_csv": str(written_csv_path),
        "max_step_columns": args.max_step_columns,
        "include_failed": args.include_failed,
        "summary": summary,
    }

    phase_report_path = package_paths.reports_dir / "csv_export_report.json"
    write_json(phase_report_path, report)
    report["csv_export_report"] = str(phase_report_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())