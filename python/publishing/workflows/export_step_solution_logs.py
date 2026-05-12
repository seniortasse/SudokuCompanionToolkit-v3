from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.step_solutions.book_loader import (
    load_book_info,
    load_book_puzzle_records,
    select_puzzle_records,
)
from python.publishing.step_solutions.locale_templates import (
    normalize_step_solution_locale,
    resolve_solution_template_paths,
)
from python.publishing.step_solutions.log_generator import (
    DEFAULT_LEGACY_ROOT,
    generate_user_logs_for_instances,
)
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
            "Generate localized Excel user_logs workbooks for a book. "
            "This is Phase 4 of the step-solution package pipeline."
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
        help="Optional maximum number of puzzles to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing user log files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip puzzles whose user log already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write legacy input JSON files and manifest only; do not call legacy generator.",
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
        skip_existing=args.skip_existing,
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

    template_paths = resolve_solution_template_paths(locale)

    results = generate_user_logs_for_instances(
        instances=instances,
        paths=package_paths,
        locale=locale,
        template_paths=template_paths,
        legacy_root=args.legacy_root,
        legacy_command=args.legacy_command,
        force=args.force,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

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

    manifest.completed_puzzle_count = sum(1 for result in results if result.status == "ok")
    manifest.failed_puzzle_count = sum(1 for result in results if result.status == "failed")

    manifest_path = write_manifest(manifest, package_paths)

    report = {
        "phase": "phase_4_user_log_generation",
        "book": {
            "book_id": book_info.book_id,
            "title": book_info.title,
            "subtitle": book_info.subtitle,
            "manifest_puzzle_count": book_info.puzzle_count,
            "loaded_puzzle_count": len(all_records),
            "selected_puzzle_count": len(selected_records),
        },
        "locale": locale,
        "dry_run": args.dry_run,
        "package": {
            "package_id": request.package_id(),
            "package_root": str(package_paths.package_root),
            "manifest_json_path": str(manifest_path),
            "user_logs_dir": str(package_paths.user_logs_dir),
            "temp_dir": str(package_paths.temp_dir),
        },
        "templates": template_paths.to_dict(),
        "results": [result.to_dict() for result in results],
        "summary": {
            "ok": sum(1 for result in results if result.status == "ok"),
            "failed": sum(1 for result in results if result.status == "failed"),
            "skipped_existing": sum(
                1 for result in results if result.status == "skipped_existing"
            ),
            "dry_run": sum(1 for result in results if result.status == "dry_run"),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())