from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from python.publishing.step_solutions.identity import (
    book_id_to_commercial_code,
    make_step_solution_identity,
)
from python.publishing.step_solutions.models import (
    StepSolutionAssetRecord,
    StepSolutionPackageManifest,
    StepSolutionPackagePaths,
    StepSolutionPackageRequest,
)
from python.publishing.step_solutions.paths import relative_to_package


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_initial_manifest(
    request: StepSolutionPackageRequest,
    paths: StepSolutionPackagePaths,
    local_puzzle_codes: Optional[Iterable[str]] = None,
    created_at: Optional[str] = None,
) -> StepSolutionPackageManifest:
    """
    Build the initial internal manifest for a package.

    At Phase 1 this is a scaffold: assets are marked as planned.
    Later phases update those assets with real workbook/image/step data.
    """

    commercial_book_code = book_id_to_commercial_code(request.book_id)

    assets = []
    for local_puzzle_code in local_puzzle_codes or []:
        identity = make_step_solution_identity(
            book_id=request.book_id,
            local_puzzle_code=local_puzzle_code,
        )
        assets.append(
            StepSolutionAssetRecord(
                book_id=identity.book_id,
                commercial_book_code=identity.commercial_book_code,
                internal_puzzle_code=identity.internal_puzzle_code,
                external_puzzle_code=identity.external_puzzle_code,
                commercial_problem_id=identity.commercial_problem_id,
            )
        )

    return StepSolutionPackageManifest(
        schema_version="step_solution_package_manifest.v1",
        package_id=request.package_id(),
        book_id=request.book_id,
        commercial_book_code=commercial_book_code,
        locale=request.locale,
        created_at=created_at or _now_iso(),
        puzzle_count=len(assets),
        completed_puzzle_count=0,
        failed_puzzle_count=0,
        naming_policy={
            "internal_puzzle_code_example": "L1-001",
            "external_puzzle_code_example": "L-1-1",
            "commercial_book_code_example": "B01",
            "commercial_problem_id_example": "B01-L-1-1",
            "user_log_filename_example": "L-1-1_user_logs.xlsx",
            "answer_image_filename_example": "B01-L-1-1_answer.png",
            "step_image_filename_example": "B01-L-1-1_step1.png",
        },
        paths={
            "package_root": str(paths.package_root),
            "user_logs_dir": relative_to_package(paths.user_logs_dir, paths),
            "image_files_dir": relative_to_package(paths.image_files_dir, paths),
            "reports_dir": relative_to_package(paths.reports_dir, paths),
            "temp_dir": relative_to_package(paths.temp_dir, paths),
            "manifest_json": relative_to_package(paths.manifest_json_path, paths),
            "sudoku_index_csv": relative_to_package(paths.sudoku_index_csv_path, paths),
            "validation_report": relative_to_package(paths.validation_report_path, paths),
            "export_summary": relative_to_package(paths.export_summary_path, paths),
        },
        assets=assets,
    )


def write_manifest(
    manifest: StepSolutionPackageManifest,
    paths: StepSolutionPackagePaths,
) -> Path:
    """
    Write manifest.json for the package.
    """

    write_json(paths.manifest_json_path, manifest.to_dict())
    return paths.manifest_json_path