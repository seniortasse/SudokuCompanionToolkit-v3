from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.step_solutions.book_loader import (
    load_book_info,
    load_book_puzzle_records,
    select_puzzle_records,
)
from python.publishing.step_solutions.identity import (
    make_answer_image_filename,
    make_step_image_filename,
    make_user_log_filename,
)
from python.publishing.step_solutions.locale_templates import (
    normalize_step_solution_locale,
    resolve_solution_template_paths,
)
from python.publishing.step_solutions.models import StepSolutionPackageRequest
from python.publishing.step_solutions.package_manifest import (
    build_initial_manifest,
    write_manifest,
)
from python.publishing.step_solutions.paths import (
    ensure_package_directories,
    resolve_package_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect book puzzle records for the step-solution package pipeline. "
            "This is a Phase 3 dry-run workflow: it loads book puzzles, resolves "
            "identity mappings, resolves templates, and can write an initial "
            "manifest scaffold. It does not generate Excel logs or images yet."
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
        help="Optional maximum number of puzzles to include.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Create package folders and write initial manifest.json.",
    )
    parser.add_argument(
        "--no-template-check",
        action="store_true",
        help="Do not require canonical template files to exist.",
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

    template_paths = resolve_solution_template_paths(
        locale=locale,
        require_exists=not args.no_template_check,
    )

    package_paths = resolve_package_paths(request)

    sample_records = selected_records[:10]
    sample = []
    for record in sample_records:
        sample.append(
            {
                "record_id": record.record_id,
                "internal_puzzle_code": record.local_puzzle_code,
                "user_log_filename": make_user_log_filename(record.local_puzzle_code),
                "answer_image_filename": make_answer_image_filename(
                    request.book_id,
                    record.local_puzzle_code,
                ),
                "step1_image_filename": make_step_image_filename(
                    request.book_id,
                    record.local_puzzle_code,
                    1,
                ),
                "section_code": record.section_code,
                "position_in_book": record.position_in_book,
                "position_in_section": record.position_in_section,
                "difficulty_label": record.difficulty_label,
                "weight": record.weight,
                "source_path": str(record.source_path),
            }
        )

    report = {
        "phase": "phase_3_book_discovery_dry_run",
        "book": {
            "book_id": book_info.book_id,
            "title": book_info.title,
            "subtitle": book_info.subtitle,
            "manifest_puzzle_count": book_info.puzzle_count,
            "loaded_puzzle_count": len(all_records),
            "selected_puzzle_count": len(selected_records),
            "grid_size": book_info.grid_size,
            "section_ids": list(book_info.section_ids),
            "manifest_path": str(book_info.manifest_path),
        },
        "locale": locale,
        "templates": template_paths.to_dict(),
        "package": {
            "package_id": request.package_id(),
            "package_root": str(package_paths.package_root),
            "user_logs_dir": str(package_paths.user_logs_dir),
            "image_files_dir": str(package_paths.image_files_dir),
            "manifest_json_path": str(package_paths.manifest_json_path),
        },
        "selection": {
            "only_puzzle": args.only_puzzle,
            "only_section": args.only_section,
            "limit": args.limit,
        },
        "sample_outputs": sample,
    }

    if args.write_manifest:
        ensure_package_directories(package_paths)
        manifest = build_initial_manifest(
            request=request,
            paths=package_paths,
            local_puzzle_codes=[
                record.local_puzzle_code for record in selected_records
            ],
        )
        manifest_path = write_manifest(manifest, package_paths)
        report["written_manifest"] = str(manifest_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())