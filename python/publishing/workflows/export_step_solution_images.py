from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.step_solutions.book_loader import (
    load_book_info,
    load_book_puzzle_records,
    select_puzzle_records,
)
from python.publishing.step_solutions.excel_image_exporter import (
    export_images_for_instances,
    write_image_export_report,
)
from python.publishing.step_solutions.locale_templates import normalize_step_solution_locale
from python.publishing.step_solutions.models import StepSolutionPackageRequest
from python.publishing.step_solutions.package_manifest import (
    build_initial_manifest,
    write_manifest,
)
from python.publishing.step_solutions.paths import (
    ensure_package_directories,
    relative_to_package,
    resolve_package_paths,
)
from python.publishing.step_solutions.puzzle_instance_adapter import (
    puzzle_records_to_instances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export answer and step PNG images from generated Excel user_logs "
            "workbooks. This is Phase 5 of the step-solution package pipeline."
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
        help="Optional maximum number of puzzles to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PNG image files.",
    )
    parser.add_argument(
        "--excel-visible",
        action="store_true",
        help="Show Excel while exporting images. Useful for debugging clipboard issues.",
    )
    parser.add_argument(
        "--clipboard-retries",
        type=int,
        default=8,
        help="Number of clipboard retry attempts per image.",
    )
    parser.add_argument(
        "--clipboard-sleep-seconds",
        type=float,
        default=0.12,
        help="Sleep duration between CopyPicture and clipboard read attempts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locale = normalize_step_solution_locale(args.locale)

    request = StepSolutionPackageRequest(
        book_id=args.book_id,
        locale=locale,
        output_root=args.output_root,
        books_root=args.books_root,
        force=args.force,
    )

    book_info = load_book_info(
        book_id=request.book_id,
        books_root=request.books_root,
    )

    all_records = load_book_puzzle_records(
        book_id=request.book_id,
        books_root=request.books_root,
    )

    selected_records = select_puzzle_records(
        records=all_records,
        only_puzzle=args.only_puzzle,
        only_section=args.only_section,
        limit=args.limit,
    )

    instances = puzzle_records_to_instances(selected_records)

    package_paths = resolve_package_paths(request)
    ensure_package_directories(package_paths)

    results = export_images_for_instances(
        instances=instances,
        paths=package_paths,
        locale=locale,
        excel_visible=args.excel_visible,
        force=args.force,
        clipboard_retries=args.clipboard_retries,
        clipboard_sleep_seconds=args.clipboard_sleep_seconds,
    )

    report_path = package_paths.reports_dir / "image_export_report.json"
    write_image_export_report(results, report_path)

    manifest = build_initial_manifest(
        request=request,
        paths=package_paths,
        local_puzzle_codes=[
            record.local_puzzle_code for record in selected_records
        ],
    )

    result_by_internal_code = {
        result.internal_puzzle_code: result
        for result in results
    }

    for asset in manifest.assets:
        result = result_by_internal_code.get(asset.internal_puzzle_code)
        if not result:
            continue

        asset.status = result.status
        asset.step_count = result.step_count
        asset.warnings.extend(result.warnings)
        asset.errors.extend(result.errors)

        if result.user_log_path.exists():
            asset.user_log_path = relative_to_package(
                result.user_log_path,
                package_paths,
            )

        if result.answer_image_path.exists():
            asset.answer_image_path = relative_to_package(
                result.answer_image_path,
                package_paths,
            )

        asset.step_image_paths = [
            relative_to_package(path, package_paths)
            for path in result.step_image_paths
            if path.exists()
        ]

    manifest.completed_puzzle_count = sum(1 for result in results if result.status == "ok")
    manifest.failed_puzzle_count = sum(1 for result in results if result.status == "failed")
    manifest.paths["image_export_report"] = relative_to_package(report_path, package_paths)

    manifest_path = write_manifest(manifest, package_paths)

    report = {
        "phase": "phase_5_excel_image_export",
        "book": {
            "book_id": book_info.book_id,
            "title": book_info.title,
            "subtitle": book_info.subtitle,
            "manifest_puzzle_count": book_info.puzzle_count,
            "loaded_puzzle_count": len(all_records),
            "selected_puzzle_count": len(selected_records),
        },
        "locale": locale,
        "package": {
            "package_id": request.package_id(),
            "package_root": str(package_paths.package_root),
            "manifest_json_path": str(manifest_path),
            "image_files_dir": str(package_paths.image_files_dir),
            "image_export_report": str(report_path),
        },
        "results": [result.to_dict() for result in results],
        "summary": {
            "ok": sum(1 for result in results if result.status == "ok"),
            "failed": sum(1 for result in results if result.status == "failed"),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())