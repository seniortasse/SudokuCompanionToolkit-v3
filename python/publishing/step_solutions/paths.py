from __future__ import annotations

from pathlib import Path

from python.publishing.step_solutions.identity import (
    make_answer_image_filename,
    make_step_image_filename,
    make_user_log_filename,
)
from python.publishing.step_solutions.models import (
    StepSolutionPackagePaths,
    StepSolutionPackageRequest,
)


def resolve_package_paths(request: StepSolutionPackageRequest) -> StepSolutionPackagePaths:
    """
    Resolve all folders and key files for a book-language step-solution package.
    """

    package_root = Path(request.output_root) / request.package_id()

    return StepSolutionPackagePaths(
        package_root=package_root,
        user_logs_dir=package_root / "user_logs",
        image_files_dir=package_root / "image_files",
        reports_dir=package_root / "reports",
        temp_dir=package_root / "temp",
        manifest_json_path=package_root / "manifest.json",
        sudoku_index_csv_path=package_root / "sudokuIndexFile.csv",
        validation_report_path=package_root / "reports" / "validation_report.json",
        export_summary_path=package_root / "reports" / "export_summary.txt",
    )


def ensure_package_directories(paths: StepSolutionPackagePaths, include_temp: bool = True) -> None:
    """
    Create the standard package folder structure.
    """

    paths.package_root.mkdir(parents=True, exist_ok=True)
    paths.user_logs_dir.mkdir(parents=True, exist_ok=True)
    paths.image_files_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    if include_temp:
        paths.temp_dir.mkdir(parents=True, exist_ok=True)
        legacy_inputs_dir(paths).mkdir(parents=True, exist_ok=True)


def user_log_path(
    paths: StepSolutionPackagePaths,
    local_puzzle_code: str,
) -> Path:
    """
    Path for one per-puzzle Excel log.

    Example:
        user_logs/L-1-1_user_logs.xlsx
    """

    return paths.user_logs_dir / make_user_log_filename(local_puzzle_code)


def answer_image_path(
    paths: StepSolutionPackagePaths,
    book_id: str,
    local_puzzle_code: str,
) -> Path:
    """
    Path for one answer image.

    Example:
        image_files/B01-L-1-1_answer.png
    """

    return paths.image_files_dir / make_answer_image_filename(
        book_id=book_id,
        local_puzzle_code=local_puzzle_code,
    )


def step_image_path(
    paths: StepSolutionPackagePaths,
    book_id: str,
    local_puzzle_code: str,
    step_number: int,
) -> Path:
    """
    Path for one solving step image.

    Example:
        image_files/B01-L-1-1_step1.png
    """

    return paths.image_files_dir / make_step_image_filename(
        book_id=book_id,
        local_puzzle_code=local_puzzle_code,
        step_number=step_number,
    )


def relative_to_package(path: Path, paths: StepSolutionPackagePaths) -> str:
    """
    Convert an absolute/package-local path to a POSIX-style path relative to package root.

    This is the form expected inside sudokuIndexFile.csv and manifest.json.

    Example:
        image_files/B01-L-1-1_step1.png
    """

    return path.relative_to(paths.package_root).as_posix()



def legacy_inputs_dir(paths: StepSolutionPackagePaths) -> Path:
    """
    Temp folder for per-puzzle JSON payloads handed to the legacy solution-log engine.

    These files are useful for debugging because they show exactly what the
    publishing platform asked the old generator to solve/render.
    """

    return paths.temp_dir / "legacy_inputs"


def legacy_input_json_path(
    paths: StepSolutionPackagePaths,
    local_puzzle_code: str,
) -> Path:
    """
    Path for one legacy-compatible puzzle input JSON file.

    Example:
        temp/legacy_inputs/L-1-1_input.json
    """

    # Reuse the user-facing puzzle code by removing the fixed Excel suffix.
    filename = make_user_log_filename(local_puzzle_code).replace(
        "_user_logs.xlsx",
        "_input.json",
    )
    return legacy_inputs_dir(paths) / filename