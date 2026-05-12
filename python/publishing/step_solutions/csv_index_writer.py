from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from python.publishing.step_solutions.identity import (
    make_answer_image_filename,
    make_step_image_filename,
    make_step_solution_identity,
)
from python.publishing.step_solutions.models import (
    StepSolutionCsvRow,
    StepSolutionPackagePaths,
)


DEFAULT_MAX_STEP_COLUMNS = 40


def csv_headers(max_step_columns: int = DEFAULT_MAX_STEP_COLUMNS) -> List[str]:
    """
    Return the external-compatible sudokuIndexFile.csv header.

    The structure intentionally preserves the legacy wide format:
        Problem ID, Problem Name, Book, Level, Answer,
        Step 1, Explanation 1, ... Step 40, Explanation 40
    """

    headers = [
        "Problem ID",
        "Problem Name",
        "Book",
        "Level",
        "Answer",
    ]

    for step_number in range(1, int(max_step_columns) + 1):
        headers.append(f"Step {step_number}")
        headers.append(f"Explanation {step_number}")

    return headers


def read_image_export_report(report_path: Path) -> Dict[str, Any]:
    """
    Read reports/image_export_report.json produced by Phase 5.
    """

    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Image export report not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def image_export_results_from_report(report_path: Path) -> List[Dict[str, Any]]:
    """
    Return image export result objects from the Phase 5 report.
    """

    payload = read_image_export_report(report_path)
    results = list(payload.get("results") or [])

    if not results:
        raise ValueError(f"No image export results found in report: {report_path}")

    return results


def write_sudoku_index_csv_from_image_report(
    report_path: Path,
    csv_path: Path,
    package_paths: StepSolutionPackagePaths,
    max_step_columns: int = DEFAULT_MAX_STEP_COLUMNS,
    include_failed: bool = False,
) -> Path:
    """
    Build sudokuIndexFile.csv from the Phase 5 image export report.

    This is the normal Phase 6 path.
    """

    results = image_export_results_from_report(report_path)
    rows = [
        row_from_image_export_result(
            result=result,
            package_paths=package_paths,
            include_failed=include_failed,
        )
        for result in results
    ]

    rows = [row for row in rows if row is not None]

    return write_sudoku_index_csv(
        rows=rows,
        csv_path=csv_path,
        max_step_columns=max_step_columns,
    )


def row_from_image_export_result(
    result: Dict[str, Any],
    package_paths: StepSolutionPackagePaths,
    include_failed: bool = False,
) -> Optional[StepSolutionCsvRow]:
    """
    Convert one Phase 5 image export result into one sudokuIndexFile.csv row.

    Important CSV identity rule:
        Problem ID and image filenames keep the commercial machine identity,
        for example B01-L-1-1.

        The Book and Level columns are user-facing labels expected by the
        downstream online tool:
            B01    -> Book 01
            L-1-1  -> Level 1
    """

    status = str(result.get("status") or "")
    if status != "ok" and not include_failed:
        return None

    book_id = str(result.get("book_id") or "")
    locale = str(result.get("locale") or "")
    internal_puzzle_code = str(result.get("internal_puzzle_code") or "")
    external_puzzle_code = str(result.get("external_puzzle_code") or "")
    commercial_book_code = str(result.get("commercial_book_code") or "")
    commercial_problem_id = str(result.get("commercial_problem_id") or "")

    if not commercial_book_code or not external_puzzle_code or not commercial_problem_id:
        identity = make_step_solution_identity(
            book_id=book_id,
            local_puzzle_code=internal_puzzle_code,
        )
        commercial_book_code = identity.commercial_book_code
        external_puzzle_code = identity.external_puzzle_code
        commercial_problem_id = identity.commercial_problem_id

    answer_path = Path(str(result.get("answer_image_path") or ""))
    step_paths = [
        Path(str(path))
        for path in list(result.get("step_image_paths") or [])
        if str(path or "").strip()
    ]
    explanations = [
        str(text or "")
        for text in list(result.get("step_explanations") or [])
    ]

    answer_rel = _normalize_package_relative_image_path(
        path=answer_path,
        package_paths=package_paths,
        fallback_filename=make_answer_image_filename(
            book_id=book_id,
            local_puzzle_code=internal_puzzle_code,
            locale=locale,
        ),
    )

    step_rels = []
    for index, step_path in enumerate(step_paths, start=1):
        step_rels.append(
            _normalize_package_relative_image_path(
                path=step_path,
                package_paths=package_paths,
                fallback_filename=make_step_image_filename(
                    book_id=book_id,
                    local_puzzle_code=internal_puzzle_code,
                    step_number=index,
                    locale=locale,
                ),
            )
        )

    return StepSolutionCsvRow(
        problem_id=_csv_problem_id(
            commercial_book_code=commercial_book_code,
            external_puzzle_code=external_puzzle_code,
            locale=locale,
            fallback_problem_id=commercial_problem_id,
        ),
        problem_name=external_puzzle_code,
        book=_csv_book_label(
            commercial_book_code=commercial_book_code,
            locale=locale,
        ),
        level=_csv_level_label(external_puzzle_code),
        answer=answer_rel,
        steps=step_rels,
        explanations=explanations,
    )


def write_sudoku_index_csv(
    rows: Iterable[StepSolutionCsvRow],
    csv_path: Path,
    max_step_columns: int = DEFAULT_MAX_STEP_COLUMNS,
) -> Path:
    """
    Write sudokuIndexFile.csv in the external-compatible wide format.

    This rewrites the CSV from the supplied row collection. It does not append.
    That is intentional so resume runs do not create duplicate/stale rows.
    """

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = csv_headers(max_step_columns=max_step_columns)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=headers,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row.to_dict(max_step_columns=max_step_columns))

    return path


def build_csv_summary(csv_path: Path) -> Dict[str, Any]:
    """
    Build a tiny summary for a generated sudokuIndexFile.csv.
    """

    path = Path(csv_path)
    if not path.exists():
        return {
            "csv_path": str(path),
            "exists": False,
            "row_count": 0,
        }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        row_count = sum(1 for _ in reader)
        headers = list(reader.fieldnames or [])

    return {
        "csv_path": str(path),
        "exists": True,
        "row_count": row_count,
        "headers": headers,
    }


def _normalize_package_relative_image_path(
    path: Path,
    package_paths: StepSolutionPackagePaths,
    fallback_filename: str,
) -> str:
    """
    Convert Phase 5 paths into CSV-compatible relative image paths.

    The external CSV should contain:
        image_files/B01-L-1-1_step1.png

    not absolute paths and not package-root paths.
    """

    raw = str(path or "").strip()

    if raw:
        path_obj = Path(raw)

        try:
            rel = path_obj.resolve().relative_to(package_paths.package_root.resolve())
            return rel.as_posix()
        except Exception:
            pass

        # If the path is already relative and includes image_files, preserve it.
        parts = path_obj.parts
        if "image_files" in parts:
            index = parts.index("image_files")
            return Path(*parts[index:]).as_posix()

        # If only a filename was provided, place it under image_files.
        if path_obj.name:
            return (Path("image_files") / path_obj.name).as_posix()

    return (Path("image_files") / fallback_filename).as_posix()


def _csv_problem_id(
    commercial_book_code: str,
    external_puzzle_code: str,
    locale: str = "",
    fallback_problem_id: str = "",
) -> str:
    """
    Build the localized CSV Problem ID expected by the online software.

    Examples:
        B01 + en + L-1-1 -> B-11101-L-1-1
        B01 + fr + L-1-1 -> B-33301-L-1-1
        B02 + de + L-2-34 -> B-22202-L-2-34

    This affects only the CSV Problem ID column. It does not change:
        - image filenames
        - package folder names
        - manifest identity
    """

    localized_book_code = _csv_localized_book_code(
        commercial_book_code=commercial_book_code,
        locale=locale,
    )
    external_code = str(external_puzzle_code or "").strip()

    if localized_book_code and external_code:
        return f"B-{localized_book_code}-{external_code}"

    return str(fallback_problem_id or "").strip()


def _csv_book_label(commercial_book_code: str, locale: str = "") -> str:
    """
    Build the localized CSV Book column value expected by the online software.

    Examples:
        B01 + en -> Book 11101
        B01 + fr -> Book 33301
        B02 + de -> Book 22202

    This affects only the CSV Book column. It does not change:
        - image filenames
        - package folder names
        - manifest identity
    """

    localized_book_code = _csv_localized_book_code(
        commercial_book_code=commercial_book_code,
        locale=locale,
    )

    if localized_book_code:
        return f"Book {localized_book_code}"

    value = str(commercial_book_code or "").strip()
    if not value:
        return ""

    return value


def _csv_localized_book_code(commercial_book_code: str, locale: str = "") -> str:
    """
    Combine numeric language code + numeric book code.

    Examples:
        en + B01 -> 11101
        de + B01 -> 22201
        fr + B01 -> 33301
        es + B01 -> 44401
        it + B01 -> 55501
    """

    language_code = _csv_language_numeric_code(locale)
    book_number = _csv_book_number_token(commercial_book_code)

    if language_code and book_number:
        return f"{language_code}{book_number}"

    return ""


def _csv_book_number_token(commercial_book_code: str) -> str:
    """
    Extract the numeric book part from a commercial book code.

    Examples:
        B01    -> 01
        B1     -> 01
        B02    -> 02
        B-2025 -> 2025
    """

    value = str(commercial_book_code or "").strip()
    if not value:
        return ""

    match = re.fullmatch(r"B(\d+)", value)
    if match:
        number = match.group(1)
        if len(number) <= 2:
            return f"{int(number):02d}"
        return number

    match = re.fullmatch(r"B-(\d+)", value)
    if match:
        return match.group(1)

    return ""


def _csv_language_numeric_code(locale: str = "") -> str:
    """
    Map package locale to the numeric language code used by the online software.

    Current convention:
        en -> 111
        de -> 222
        fr -> 333
        es -> 444
        it -> 555
    """

    value = str(locale or "").strip().lower().replace("_", "-")
    language = value.split("-", 1)[0]

    return {
        "en": "111",
        "de": "222",
        "fr": "333",
        "es": "444",
        "it": "555",
    }.get(language, "")


def _csv_level_label(external_puzzle_code: str) -> str:
    """
    Convert the external puzzle code into the user-facing CSV Level label.

    Examples:
        L-1-1   -> Level 1
        L-2-48  -> Level 2
        L1-001  -> Level 1
        L2-034  -> Level 2
    """

    value = str(external_puzzle_code or "").strip()
    if not value:
        return ""

    match = re.match(r"^L-(\d+)-", value)
    if match:
        return f"Level {int(match.group(1))}"

    match = re.match(r"^L(\d+)-", value)
    if match:
        return f"Level {int(match.group(1))}"

    return value


def _level_from_external_puzzle_code(external_puzzle_code: str) -> str:
    """
    Legacy helper retained for backward compatibility.

    Old behavior:
        L-1-1 -> L-1
        L-2-34 -> L-2

    New CSV output should use _csv_level_label instead:
        L-1-1 -> Level 1
        L-2-34 -> Level 2
    """

    value = str(external_puzzle_code or "").strip()
    parts = value.split("-")

    if len(parts) >= 3 and parts[0] == "L":
        return f"L-{parts[1]}"

    return ""