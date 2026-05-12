from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from python.publishing.step_solutions.book_loader import (
    load_book_info,
    load_book_puzzle_records,
    select_puzzle_records,
)
from python.publishing.step_solutions.csv_index_writer import (
    DEFAULT_MAX_STEP_COLUMNS,
    build_csv_summary,
    write_sudoku_index_csv_from_image_report,
)
from python.publishing.step_solutions.excel_image_exporter import (
    export_images_for_instances,
    write_image_export_report,
)
from python.publishing.step_solutions.locale_templates import (
    normalize_step_solution_locale,
    resolve_solution_template_paths,
)
from python.publishing.step_solutions.log_generator import (
    DEFAULT_LEGACY_ROOT,
    generate_user_logs_for_instances,
)
from python.publishing.step_solutions.models import (
    StepSolutionPackageExportResult,
    StepSolutionPackageRequest,
)
from python.publishing.step_solutions.package_manifest import (
    build_initial_manifest,
    write_json,
    write_manifest,
)
from python.publishing.step_solutions.paths import (
    ensure_package_directories,
    relative_to_package,
    resolve_package_paths,
)
from python.publishing.step_solutions.progress import (
    ProgressTimer,
    print_progress,
)
from python.publishing.step_solutions.puzzle_instance_adapter import (
    puzzle_records_to_instances,
)


def export_step_solution_package(
    book_id: str,
    locale: str,
    books_root: Path = Path("datasets/sudoku_books/classic9/books"),
    output_root: Path = Path("datasets/sudoku_books/classic9/step_solution_packages"),
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    legacy_command: Optional[str] = None,
    only_puzzle: Optional[str] = None,
    only_section: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
    skip_existing: bool = False,
    excel_visible: bool = False,
    max_step_columns: int = DEFAULT_MAX_STEP_COLUMNS,
    include_failed_csv_rows: bool = False,
    logs_only: bool = False,
    images_only: bool = False,
    csv_only: bool = False,
    dry_run_logs: bool = False,
    clipboard_retries: int = 8,
    clipboard_sleep_seconds: float = 0.12,
) -> StepSolutionPackageExportResult:
    """
    Export one complete localized step-solution package.

    Normal mode runs:
        logs -> images -> sudokuIndexFile.csv

    Development modes:
        logs_only
        images_only
        csv_only
        dry_run_logs
    """

    locale = normalize_step_solution_locale(locale)

    request = StepSolutionPackageRequest(
        book_id=book_id,
        locale=locale,
        books_root=books_root,
        output_root=output_root,
        force=force,
        skip_existing=skip_existing,
    )

    package_paths = resolve_package_paths(request)
    ensure_package_directories(package_paths)

    package_report_path = package_paths.reports_dir / "package_export_report.json"
    image_report_path = package_paths.reports_dir / "image_export_report.json"
    csv_report_path = package_paths.reports_dir / "csv_export_report.json"

    warnings: List[str] = []
    errors: List[str] = []

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
        only_puzzle=only_puzzle,
        only_section=only_section,
        limit=limit,
    )

    instances = puzzle_records_to_instances(selected_records)

    template_paths = resolve_solution_template_paths(locale)

    package_timer = ProgressTimer.start()
    selected_puzzle_count = len(selected_records)

    print_progress(
        "PACKAGE",
        (
            f"{request.book_id}/{locale} started | "
            f"puzzles={selected_puzzle_count} | "
            f"force={force} | skip_existing={skip_existing} | "
            f"package={package_paths.package_root}"
        ),
    )

    log_results = []
    image_results = []
    csv_path = package_paths.sudoku_index_csv_path

    if csv_only:
        logs_phase_status = "skipped_csv_only"
        images_phase_status = "skipped_csv_only"
    else:
        logs_phase_status = "pending"
        images_phase_status = "pending"

    if not images_only and not csv_only:
        logs_timer = ProgressTimer.start()
        print_progress(
            "LOGS",
            f"{request.book_id}/{locale} started | puzzles={selected_puzzle_count}",
        )

        log_results = generate_user_logs_for_instances(
            instances=instances,
            paths=package_paths,
            locale=locale,
            template_paths=template_paths,
            legacy_root=legacy_root,
            legacy_command=legacy_command,
            force=force,
            skip_existing=skip_existing,
            dry_run=dry_run_logs,
        )
        logs_phase_status = "done"

        log_ok_now = sum(1 for result in log_results if result.status == "ok")
        log_failed_now = sum(1 for result in log_results if result.status == "failed")
        log_skipped_now = sum(
            1 for result in log_results if result.status == "skipped_existing"
        )
        log_dry_run_now = sum(1 for result in log_results if result.status == "dry_run")

        print_progress(
            "LOGS",
            (
                f"{request.book_id}/{locale} done | "
                f"ok={log_ok_now} | failed={log_failed_now} | "
                f"skipped={log_skipped_now} | dry_run={log_dry_run_now} | "
                f"elapsed={logs_timer.elapsed_text()}"
            ),
        )

        if dry_run_logs:
            warnings.append(
                "dry_run_logs=True: user logs were not generated; image and CSV phases were skipped."
            )
            images_phase_status = "skipped_dry_run_logs"

    elif images_only:
        logs_phase_status = "skipped_images_only"

    if not logs_only and not csv_only and not dry_run_logs:
        images_timer = ProgressTimer.start()
        print_progress(
            "IMAGES",
            f"{request.book_id}/{locale} started | puzzles={selected_puzzle_count}",
        )

        image_results = export_images_for_instances(
            instances=instances,
            paths=package_paths,
            locale=locale,
            excel_visible=excel_visible,
            force=force,
            clipboard_retries=clipboard_retries,
            clipboard_sleep_seconds=clipboard_sleep_seconds,
        )
        write_image_export_report(image_results, image_report_path)
        images_phase_status = "done"

        image_ok_now = sum(1 for result in image_results if result.status == "ok")
        image_failed_now = sum(1 for result in image_results if result.status == "failed")

        print_progress(
            "IMAGES",
            (
                f"{request.book_id}/{locale} done | "
                f"ok={image_ok_now} | failed={image_failed_now} | "
                f"elapsed={images_timer.elapsed_text()}"
            ),
        )

    elif logs_only:
        images_phase_status = "skipped_logs_only"

    if not logs_only and not images_only and not dry_run_logs:
        csv_timer = ProgressTimer.start()
        print_progress(
            "CSV",
            f"{request.book_id}/{locale} started | output={package_paths.sudoku_index_csv_path}",
        )

        csv_path = write_sudoku_index_csv_from_image_report(
            report_path=image_report_path,
            csv_path=package_paths.sudoku_index_csv_path,
            package_paths=package_paths,
            max_step_columns=max_step_columns,
            include_failed=include_failed_csv_rows,
        )
        csv_summary = build_csv_summary(csv_path)

        print_progress(
            "CSV",
            (
                f"{request.book_id}/{locale} done | "
                f"rows={csv_summary.get('row_count', 0)} | "
                f"elapsed={csv_timer.elapsed_text()}"
            ),
        )

        csv_report = {
            "phase": "phase_7_package_csv_substep",
            "book_id": request.book_id,
            "locale": locale,
            "package_id": request.package_id(),
            "image_export_report": str(image_report_path),
            "sudoku_index_csv": str(csv_path),
            "max_step_columns": max_step_columns,
            "include_failed": include_failed_csv_rows,
            "summary": csv_summary,
        }
        write_json(csv_report_path, csv_report)
        csv_phase_status = "done"
    else:
        csv_summary = build_csv_summary(csv_path)
        csv_phase_status = "skipped"

    manifest = build_initial_manifest(
        request=request,
        paths=package_paths,
        local_puzzle_codes=[
            record.local_puzzle_code for record in selected_records
        ],
    )

    log_by_code = {
        result.internal_puzzle_code: result
        for result in log_results
    }
    image_by_code = {
        result.internal_puzzle_code: result
        for result in image_results
    }

    for asset in manifest.assets:
        log_result = log_by_code.get(asset.internal_puzzle_code)
        image_result = image_by_code.get(asset.internal_puzzle_code)

        if log_result is not None:
            asset.status = log_result.status
            asset.step_count = log_result.step_count
            asset.warnings.extend(log_result.warnings)
            asset.errors.extend(log_result.errors)

            if log_result.user_log_path.exists():
                asset.user_log_path = relative_to_package(
                    log_result.user_log_path,
                    package_paths,
                )

        if image_result is not None:
            asset.status = image_result.status
            asset.step_count = image_result.step_count
            asset.warnings.extend(image_result.warnings)
            asset.errors.extend(image_result.errors)

            if image_result.user_log_path.exists():
                asset.user_log_path = relative_to_package(
                    image_result.user_log_path,
                    package_paths,
                )

            if image_result.answer_image_path.exists():
                asset.answer_image_path = relative_to_package(
                    image_result.answer_image_path,
                    package_paths,
                )

            asset.step_image_paths = [
                relative_to_package(path, package_paths)
                for path in image_result.step_image_paths
                if path.exists()
            ]

    if image_results:
        manifest.completed_puzzle_count = sum(
            1 for result in image_results if result.status == "ok"
        )
        manifest.failed_puzzle_count = sum(
            1 for result in image_results if result.status == "failed"
        )
    elif log_results:
        manifest.completed_puzzle_count = sum(
            1 for result in log_results if result.status in ("ok", "skipped_existing")
        )
        manifest.failed_puzzle_count = sum(
            1 for result in log_results if result.status == "failed"
        )

    manifest.paths["image_export_report"] = relative_to_package(
        image_report_path,
        package_paths,
    )
    manifest.paths["csv_export_report"] = relative_to_package(
        csv_report_path,
        package_paths,
    )
    manifest.paths["package_export_report"] = relative_to_package(
        package_report_path,
        package_paths,
    )

    manifest_path = write_manifest(manifest, package_paths)

    log_ok_count = sum(1 for result in log_results if result.status == "ok")
    log_failed_count = sum(1 for result in log_results if result.status == "failed")
    log_skipped_count = sum(
        1 for result in log_results if result.status == "skipped_existing"
    )

    image_ok_count = sum(1 for result in image_results if result.status == "ok")
    image_failed_count = sum(1 for result in image_results if result.status == "failed")

    if log_failed_count or image_failed_count:
        status = "failed"
        if log_failed_count:
            errors.append(f"{log_failed_count} user log generation result(s) failed.")
        if image_failed_count:
            errors.append(f"{image_failed_count} image export result(s) failed.")
    elif dry_run_logs:
        status = "dry_run"
    else:
        status = "ok"

    result = StepSolutionPackageExportResult(
        book_id=request.book_id,
        locale=locale,
        package_id=request.package_id(),
        package_root=package_paths.package_root,
        selected_puzzle_count=len(selected_records),
        log_ok_count=log_ok_count,
        log_failed_count=log_failed_count,
        log_skipped_count=log_skipped_count,
        image_ok_count=image_ok_count,
        image_failed_count=image_failed_count,
        csv_path=csv_path if csv_path.exists() else None,
        manifest_path=manifest_path,
        package_report_path=package_report_path,
        status=status,
        warnings=warnings,
        errors=errors,
    )

    package_report = {
        "phase": "phase_7_step_solution_package_export",
        "status": status,
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
            "user_logs_dir": str(package_paths.user_logs_dir),
            "image_files_dir": str(package_paths.image_files_dir),
            "sudoku_index_csv": str(csv_path),
            "image_export_report": str(image_report_path),
            "csv_export_report": str(csv_report_path),
            "package_export_report": str(package_report_path),
        },
        "selection": {
            "only_puzzle": only_puzzle,
            "only_section": only_section,
            "limit": limit,
        },
        "options": {
            "force": force,
            "skip_existing": skip_existing,
            "excel_visible": excel_visible,
            "max_step_columns": max_step_columns,
            "include_failed_csv_rows": include_failed_csv_rows,
            "logs_only": logs_only,
            "images_only": images_only,
            "csv_only": csv_only,
            "dry_run_logs": dry_run_logs,
        },
        "phases": {
            "logs": logs_phase_status,
            "images": images_phase_status,
            "csv": csv_phase_status,
        },
        "summary": result.to_dict(),
        "csv_summary": csv_summary,
        "log_results": [result.to_dict() for result in log_results],
        "image_results": [result.to_dict() for result in image_results],
        "warnings": warnings,
        "errors": errors,
    }

    write_json(package_report_path, package_report)

    print_progress(
        "PACKAGE",
        (
            f"{request.book_id}/{locale} {status.upper()} | "
            f"logs_ok={log_ok_count} | logs_failed={log_failed_count} | "
            f"images_ok={image_ok_count} | images_failed={image_failed_count} | "
            f"elapsed={package_timer.elapsed_text()}"
        ),
    )

    return result