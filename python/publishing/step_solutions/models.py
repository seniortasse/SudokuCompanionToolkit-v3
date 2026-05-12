from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SUPPORTED_STEP_SOLUTION_LOCALES = ("en", "fr", "de", "it", "es")


@dataclass(frozen=True)
class StepSolutionPackageRequest:
    """
    Request to build one localized step-solution package.
    """

    book_id: str
    locale: str
    output_root: Path = Path("datasets/sudoku_books/classic9/step_solution_packages")
    books_root: Path = Path("datasets/sudoku_books/classic9/books")
    force: bool = False
    skip_existing: bool = False
    keep_temp: bool = False

    def package_id(self) -> str:
        return f"{self.book_id}-{self.locale}"


@dataclass(frozen=True)
class StepSolutionPackagePaths:
    """
    Concrete folder and file paths for one book-language package.
    """

    package_root: Path
    user_logs_dir: Path
    image_files_dir: Path
    reports_dir: Path
    temp_dir: Path
    manifest_json_path: Path
    sudoku_index_csv_path: Path
    validation_report_path: Path
    export_summary_path: Path


@dataclass
class StepSolutionAssetRecord:
    """
    Manifest trace record for one puzzle in a generated package.
    """

    book_id: str
    commercial_book_code: str
    internal_puzzle_code: str
    external_puzzle_code: str
    commercial_problem_id: str

    user_log_path: Optional[str] = None
    answer_image_path: Optional[str] = None
    step_image_paths: List[str] = field(default_factory=list)

    step_count: int = 0
    status: str = "planned"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepSolutionPackageManifest:
    """
    Internal JSON manifest for one generated solution package.
    """

    schema_version: str
    package_id: str
    book_id: str
    commercial_book_code: str
    locale: str
    created_at: str

    puzzle_count: int = 0
    completed_puzzle_count: int = 0
    failed_puzzle_count: int = 0

    naming_policy: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    assets: List[StepSolutionAssetRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["assets"] = [asset.to_dict() for asset in self.assets]
        return payload


@dataclass(frozen=True)
class StepSolutionBookInfo:
    """
    Minimal book-level information needed by the step-solution package pipeline.
    """

    book_id: str
    title: str
    subtitle: str
    puzzle_count: int
    grid_size: int
    section_ids: Sequence[str]
    manifest_path: Path
    book_dir: Path

    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        payload["book_dir"] = str(self.book_dir)
        payload["section_ids"] = list(self.section_ids)
        return payload


@dataclass(frozen=True)
class StepSolutionPuzzleRecord:
    """
    Normalized puzzle record loaded from a book's puzzles/ folder.
    """

    record_id: str
    book_id: str
    section_id: str
    section_code: str
    local_puzzle_code: str
    friendly_puzzle_id: str

    givens81: str
    solution81: str

    position_in_book: int
    position_in_section: int

    difficulty_label: str
    puzzle_difficulty: str
    weight: Optional[int]
    technique_count: Optional[int]
    techniques_used: List[str]
    technique_histogram: Dict[str, Any]

    source_path: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        return payload


@dataclass(frozen=True)
class StepSolutionPuzzleInstance:
    """
    Adapter shape used by log generation and image export phases.
    """

    book_id: str
    internal_puzzle_code: str
    external_puzzle_code: str
    commercial_book_code: str
    commercial_problem_id: str

    givens81: str
    solution81: str
    grid_size: int = 9

    section_code: str = ""
    position_in_book: int = 0
    position_in_section: int = 0
    difficulty_label: str = ""
    weight: Optional[int] = None

    source_record_id: str = ""
    source_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.source_path is not None:
            payload["source_path"] = str(self.source_path)
        return payload


@dataclass(frozen=True)
class StepSolutionLogGenerationResult:
    """
    Result of generating one localized per-puzzle Excel user log.
    """

    book_id: str
    locale: str

    internal_puzzle_code: str
    external_puzzle_code: str
    commercial_book_code: str
    commercial_problem_id: str

    user_log_path: Path
    legacy_input_path: Optional[Path] = None

    status: str = "planned"
    step_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.status == "ok" and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["user_log_path"] = str(self.user_log_path)
        if self.legacy_input_path is not None:
            payload["legacy_input_path"] = str(self.legacy_input_path)
        return payload


@dataclass(frozen=True)
class StepSolutionImageExportResult:
    """
    Result of exporting answer and step images from one Excel user log workbook.
    """

    book_id: str
    locale: str

    internal_puzzle_code: str
    external_puzzle_code: str
    commercial_book_code: str
    commercial_problem_id: str

    user_log_path: Path
    answer_image_path: Path
    step_image_paths: List[Path] = field(default_factory=list)
    step_explanations: List[str] = field(default_factory=list)

    step_count: int = 0
    status: str = "planned"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.status == "ok" and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["user_log_path"] = str(self.user_log_path)
        payload["answer_image_path"] = str(self.answer_image_path)
        payload["step_image_paths"] = [str(path) for path in self.step_image_paths]
        return payload
    

@dataclass(frozen=True)
class StepSolutionCsvRow:
    """
    One row in the external-compatible sudokuIndexFile.csv.

    This is intentionally external/user-facing:
        - commercial book code, e.g. B01
        - external puzzle code, e.g. L-1-1
        - commercial problem id, e.g. B01-L-1-1
        - relative image paths under image_files/
    """

    problem_id: str
    problem_name: str
    book: str
    level: str
    answer: str
    steps: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self, max_step_columns: int = 40) -> Dict[str, str]:
        row: Dict[str, str] = {
            "Problem ID": self.problem_id,
            "Problem Name": self.problem_name,
            "Book": self.book,
            "Level": self.level,
            "Answer": self.answer,
        }

        for step_number in range(1, int(max_step_columns) + 1):
            index = step_number - 1

            row[f"Step {step_number}"] = (
                self.steps[index] if index < len(self.steps) else ""
            )
            row[f"Explanation {step_number}"] = (
                self.explanations[index] if index < len(self.explanations) else ""
            )

        return row
    

@dataclass(frozen=True)
class StepSolutionPackageExportResult:
    """
    End-to-end export result for one book-language package.

    This is produced by the unified Phase 7 package workflow:
        logs -> images -> sudokuIndexFile.csv -> package report
    """

    book_id: str
    locale: str
    package_id: str
    package_root: Path

    selected_puzzle_count: int = 0

    log_ok_count: int = 0
    log_failed_count: int = 0
    log_skipped_count: int = 0

    image_ok_count: int = 0
    image_failed_count: int = 0

    csv_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    package_report_path: Optional[Path] = None

    status: str = "planned"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.status == "ok" and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["package_root"] = str(self.package_root)
        if self.csv_path is not None:
            payload["csv_path"] = str(self.csv_path)
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
        if self.package_report_path is not None:
            payload["package_report_path"] = str(self.package_report_path)
        return payload