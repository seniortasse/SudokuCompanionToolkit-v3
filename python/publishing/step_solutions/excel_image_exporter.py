from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from python.publishing.step_solutions.identity import (
    make_answer_image_filename,
    make_step_image_filename,
)
from python.publishing.step_solutions.models import (
    StepSolutionImageExportResult,
    StepSolutionPackagePaths,
    StepSolutionPuzzleInstance,
)
from python.publishing.step_solutions.paths import (
    user_log_path,
)
from python.publishing.step_solutions.progress import (
    ProgressTimer,
    print_progress,
)

# Excel COM constants.
#
# We use raw integer values instead of win32com.client.constants so the exporter
# does not depend on makepy/gencache wrapper generation.
EXCEL_XL_PART = 2
EXCEL_XL_BY_ROWS = 1
EXCEL_XL_PREVIOUS = 2
EXCEL_XL_BITMAP = 2


def get_cell_coordinates_by_step_number(step_number: int) -> Dict[str, int]:
    """
    Return the Excel cell range coordinates for a step snapshot.

    This is the publishing-platform version of the legacy SudokuParser2.py
    coordinate logic.

    The workbook layout places four step snapshots per page-like block:
        step 1: top-left
        step 2: top-right
        step 3: bottom-left
        step 4: bottom-right

    Each copied step image includes:
        - black STEP X header
        - 9x9 Sudoku grid
    """

    step_number = int(step_number)
    if step_number < 1:
        raise ValueError(f"Step number must be >= 1; got {step_number!r}.")

    zero_based = step_number - 1

    page_number = zero_based // 4
    row_number_on_page = (zero_based % 4) // 2
    column_number_on_page = (zero_based % 4) % 2

    xls_page_start_row = (28 * (page_number + 1)) + 1
    xls_step_start_row = xls_page_start_row + ((13 * row_number_on_page) + 2)
    xls_step_stop_row = xls_step_start_row + 9
    xls_step_start_col = 2 + (10 * column_number_on_page)
    xls_step_stop_col = 10 + (10 * column_number_on_page)

    return {
        "up": xls_step_start_row,
        "down": xls_step_stop_row,
        "left": xls_step_start_col,
        "right": xls_step_stop_col,
    }


def rgb_to_excel_int(rgb: tuple[int, int, int]) -> int:
    """
    Convert RGB to the integer color format used by Excel COM.
    """

    return rgb[0] + (rgb[1] * 256) + (rgb[2] * 256 * 256)


def export_images_for_instance(
    instance: StepSolutionPuzzleInstance,
    paths: StepSolutionPackagePaths,
    locale: str,
    excel_visible: bool = False,
    force: bool = False,
    clipboard_retries: int = 8,
    clipboard_sleep_seconds: float = 0.12,
) -> StepSolutionImageExportResult:
    """
    Export one puzzle's answer image and all step images from its user log workbook.

    Input:
        user_logs/L-1-1_user_logs.xlsx

    Outputs:
        image_files/B01-L-1-1_answer.png
        image_files/B01-L-1-1_step1.png
        image_files/B01-L-1-1_step2.png
        ...
    """

    expected_user_log_path = user_log_path(
        paths=paths,
        local_puzzle_code=instance.internal_puzzle_code,
    )
    expected_answer_path = paths.image_files_dir / make_answer_image_filename(
        book_id=instance.book_id,
        local_puzzle_code=instance.internal_puzzle_code,
        locale=locale,
    )

    base = {
        "book_id": instance.book_id,
        "locale": locale,
        "internal_puzzle_code": instance.internal_puzzle_code,
        "external_puzzle_code": instance.external_puzzle_code,
        "commercial_book_code": instance.commercial_book_code,
        "commercial_problem_id": instance.commercial_problem_id,
        "user_log_path": expected_user_log_path,
        "answer_image_path": expected_answer_path,
    }

    if not expected_user_log_path.exists():
        return StepSolutionImageExportResult(
            **base,
            status="failed",
            errors=[f"User log workbook not found: {expected_user_log_path}"],
        )

    if expected_answer_path.exists() and not force:
        # We still do not know whether all step images exist without opening
        # the workbook, so this is not treated as a skip. The exporter will
        # overwrite only if force=True, otherwise existing files are left as-is.
        pass

    try:
        return _export_images_with_excel_com(
            instance=instance,
            paths=paths,
            locale=locale,
            user_log_xlsx_path=expected_user_log_path,
            answer_png_path=expected_answer_path,
            excel_visible=excel_visible,
            force=force,
            clipboard_retries=clipboard_retries,
            clipboard_sleep_seconds=clipboard_sleep_seconds,
        )
    except Exception as exc:
        return StepSolutionImageExportResult(
            **base,
            status="failed",
            errors=[str(exc)],
        )


def export_images_for_instances(
    instances: Iterable[StepSolutionPuzzleInstance],
    paths: StepSolutionPackagePaths,
    locale: str,
    excel_visible: bool = False,
    force: bool = False,
    clipboard_retries: int = 8,
    clipboard_sleep_seconds: float = 0.12,
) -> List[StepSolutionImageExportResult]:
    """
    Export images for many puzzle user logs.

    Emits one compact progress line per puzzle so long Excel/COM image export
    runs are observable from the terminal.
    """

    instance_list = list(instances)
    total = len(instance_list)

    results: List[StepSolutionImageExportResult] = []

    for index, instance in enumerate(instance_list, start=1):
        timer = ProgressTimer.start()

        expected_answer_path = paths.image_files_dir / make_answer_image_filename(
            book_id=instance.book_id,
            local_puzzle_code=instance.internal_puzzle_code,
            locale=locale,
        )

        print_progress(
            f"IMG {index}/{total}",
            (
                f"{instance.book_id}/{locale} "
                f"{instance.internal_puzzle_code} -> "
                f"{expected_answer_path.name} started"
            ),
        )

        result = export_images_for_instance(
            instance=instance,
            paths=paths,
            locale=locale,
            excel_visible=excel_visible,
            force=force,
            clipboard_retries=clipboard_retries,
            clipboard_sleep_seconds=clipboard_sleep_seconds,
        )
        results.append(result)

        if result.status == "ok":
            detail = f"answer + {result.step_count} steps"
        elif result.errors:
            detail = f"error={_short_progress_error(result.errors[0])}"
        else:
            detail = "no detail"

        print_progress(
            f"IMG {index}/{total}",
            (
                f"{instance.book_id}/{locale} "
                f"{instance.internal_puzzle_code} -> "
                f"{result.status.upper()} | {detail} | "
                f"elapsed={timer.elapsed_text()}"
            ),
        )

    return results


def _short_progress_error(message: str, max_length: int = 180) -> str:
    """
    Keep console progress errors readable.
    """

    text = " ".join(str(message or "").split())
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."

def write_image_export_report(
    results: List[StepSolutionImageExportResult],
    report_path: Path,
) -> Path:
    """
    Write a JSON report for image export results.
    """

    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "step_solution_image_export_report.v1",
        "result_count": len(results),
        "ok": sum(1 for result in results if result.status == "ok"),
        "failed": sum(1 for result in results if result.status == "failed"),
        "results": [result.to_dict() for result in results],
    }

    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def _export_images_with_excel_com(
    instance: StepSolutionPuzzleInstance,
    paths: StepSolutionPackagePaths,
    locale: str,
    user_log_xlsx_path: Path,
    answer_png_path: Path,
    excel_visible: bool,
    force: bool,
    clipboard_retries: int,
    clipboard_sleep_seconds: float,
) -> StepSolutionImageExportResult:
    """
    Windows/Excel implementation.

    Requires:
        pip package: pywin32
        Microsoft Excel installed

    This intentionally mirrors the old sudokuparser screenshot behavior.
    """

    try:
        import win32com.client as w3c
        from PIL import ImageGrab
    except Exception as exc:
        raise RuntimeError(
            "Excel image export requires Windows, Microsoft Excel, pywin32, "
            "and Pillow ImageGrab support."
        ) from exc

    paths.image_files_dir.mkdir(parents=True, exist_ok=True)

    xls_app = None
    xls_wb = None
    xls_ws = None

    step_paths: List[Path] = []
    explanations: List[str] = []

    try:
        # Use late-bound COM instead of gencache.EnsureDispatch.
        #
        # EnsureDispatch can fail with:
        #   "This COM object can not automate the makepy process"
        #
        # DispatchEx starts a fresh Excel instance and avoids requiring makepy
        # generated wrappers. This is more reliable for batch exports.
        xls_app = w3c.DispatchEx("Excel.Application")
        xls_app.Visible = bool(excel_visible)
        xls_app.DisplayAlerts = False

        xls_wb = _open_workbook(xls_app, user_log_xlsx_path)
        if xls_wb is None:
            raise RuntimeError(f"Could not open workbook: {user_log_xlsx_path}")

        xls_ws = xls_wb.Worksheets("Steps")

        try:
            xls_wb.AutoSaveOn = False
        except Exception:
            pass

        max_step_number = _find_max_step_number(w3c, xls_ws)

        for step_number in range(1, max_step_number + 1):
            coords = get_cell_coordinates_by_step_number(step_number)
            output_path = paths.image_files_dir / make_step_image_filename(
                book_id=instance.book_id,
                local_puzzle_code=instance.internal_puzzle_code,
                step_number=step_number,
                locale=locale,
            )

            if force or not output_path.exists():
                _copy_excel_range_as_png(
                    w3c=w3c,
                    xls_ws=xls_ws,
                    row_from=coords["up"],
                    col_from=coords["left"],
                    row_to=coords["down"],
                    col_to=coords["right"],
                    output_path=output_path,
                    image_grab=ImageGrab,
                    clipboard_retries=clipboard_retries,
                    clipboard_sleep_seconds=clipboard_sleep_seconds,
                )

            explanation_text = str(
                xls_ws.Cells(coords["up"] + 10, coords["left"]).Text or ""
            ).strip()

            step_paths.append(output_path)
            explanations.append(explanation_text)

        # Answer image:
        # Use the final solved step, but remove step highlighting and header.
        answer_coords = get_cell_coordinates_by_step_number(max_step_number)
        answer_range = xls_ws.Range(
            xls_ws.Cells(answer_coords["up"] + 1, answer_coords["left"]),
            xls_ws.Cells(answer_coords["down"], answer_coords["right"]),
        )

        answer_range.Interior.Color = rgb_to_excel_int((255, 255, 255))
        answer_range.Font.Color = rgb_to_excel_int((0, 0, 0))

        if force or not answer_png_path.exists():
            _copy_excel_range_as_png(
                w3c=w3c,
                xls_ws=xls_ws,
                row_from=answer_coords["up"] + 1,
                col_from=answer_coords["left"],
                row_to=answer_coords["down"],
                col_to=answer_coords["right"],
                output_path=answer_png_path,
                image_grab=ImageGrab,
                clipboard_retries=clipboard_retries,
                clipboard_sleep_seconds=clipboard_sleep_seconds,
            )

        return StepSolutionImageExportResult(
            book_id=instance.book_id,
            locale=locale,
            internal_puzzle_code=instance.internal_puzzle_code,
            external_puzzle_code=instance.external_puzzle_code,
            commercial_book_code=instance.commercial_book_code,
            commercial_problem_id=instance.commercial_problem_id,
            user_log_path=user_log_xlsx_path,
            answer_image_path=answer_png_path,
            step_image_paths=step_paths,
            step_explanations=explanations,
            step_count=max_step_number,
            status="ok",
        )

    finally:
        if xls_wb is not None:
            try:
                xls_wb.Close(SaveChanges=False)
            except Exception:
                pass

        if xls_app is not None:
            try:
                xls_app.Quit()
            except Exception:
                pass


def _open_workbook(xls_app, xlsx_path: Path):
    """
    Open an Excel workbook through COM.
    """

    xlsx_path_abs = str(Path(xlsx_path).resolve())

    try:
        return xls_app.Workbooks.Open(
            Filename=xlsx_path_abs,
            ReadOnly=False,
            IgnoreReadOnlyRecommended=True,
        )
    except Exception:
        try:
            return xls_app.Workbooks(xlsx_path_abs)
        except Exception:
            return None


def _find_max_step_number(w3c, xls_ws) -> int:
    """
    Find the last solving step in the Steps sheet.

    Important:
        Do not rely only on the English word "STEP".

    Localized workbooks may use headers such as:
        STEP 1
        ÉTAPE 1
        SCHRITT 1
        PASSO 1
        PASO 1

    The workbook layout is stable, so the most robust method is to scan the
    expected header cells for step numbers using get_cell_coordinates_by_step_number().
    """

    # Fast path for old English-only workbooks.
    try:
        max_step_text = xls_ws.Cells.Find(
            What="STEP",
            LookAt=EXCEL_XL_PART,
            SearchOrder=EXCEL_XL_BY_ROWS,
            SearchDirection=EXCEL_XL_PREVIOUS,
            MatchCase=False,
            SearchFormat=False,
        )

        if max_step_text is not None:
            parsed = _parse_step_number_from_text(str(max_step_text.Value or ""))
            if parsed is not None:
                return parsed
    except Exception:
        # Fall through to language-neutral layout scan.
        pass

    # Language-neutral path:
    # Scan the expected step-header cells. Stop after several consecutive
    # missing headers once at least one step has been found.
    max_found = 0
    missing_streak = 0
    max_steps_safety = 300

    for step_number in range(1, max_steps_safety + 1):
        coords = get_cell_coordinates_by_step_number(step_number)

        candidates = [
            xls_ws.Cells(coords["up"], coords["left"]),
            xls_ws.Cells(coords["up"], coords["left"] + 1),
            xls_ws.Cells(coords["up"], coords["right"]),
        ]

        header_texts = []
        for cell in candidates:
            try:
                header_texts.append(str(cell.Text or "").strip())
            except Exception:
                try:
                    header_texts.append(str(cell.Value or "").strip())
                except Exception:
                    pass

        parsed_numbers = [
            number
            for text in header_texts
            for number in [_parse_step_number_from_text(text)]
            if number is not None
        ]

        if step_number in parsed_numbers:
            max_found = step_number
            missing_streak = 0
            continue

        # Some merged headers may expose only text without the expected number
        # in a neighboring cell. Treat any non-empty header-like text on the
        # expected row as present, but prefer exact numeric matches above.
        if any(_looks_like_step_header_text(text) for text in header_texts):
            max_found = step_number
            missing_streak = 0
            continue

        if max_found > 0:
            missing_streak += 1
            if missing_streak >= 4:
                break

    if max_found > 0:
        return max_found

    raise RuntimeError(
        "Could not find any solving step header in workbook Steps sheet. "
        "Checked English STEP search and language-neutral layout scan."
    )


def _parse_step_number_from_text(text: str) -> Optional[int]:
    """
    Extract the final integer from a localized step header.

    Examples:
        STEP 13     -> 13
        ÉTAPE 13    -> 13
        SCHRITT 13  -> 13
        PASSO 13    -> 13
        PASO 13     -> 13
    """

    value = str(text or "").strip()
    if not value:
        return None

    numbers = re.findall(r"\d+", value)
    if not numbers:
        return None

    return int(numbers[-1])


def _looks_like_step_header_text(text: str) -> bool:
    """
    Heuristic for localized step headers.

    We intentionally keep this permissive because the exact language is less
    important than the stable workbook layout.
    """

    value = str(text or "").strip()
    if not value:
        return False

    normalized = (
        value.upper()
        .replace("É", "E")
        .replace("È", "E")
        .replace("Ê", "E")
        .replace("À", "A")
        .replace("Á", "A")
        .replace("Ì", "I")
        .replace("Í", "I")
        .replace("Ò", "O")
        .replace("Ó", "O")
        .replace("Ù", "U")
        .replace("Ú", "U")
    )

    keywords = (
        "STEP",
        "ETAPE",
        "SCHRITT",
        "PASSO",
        "PASO",
    )

    return any(keyword in normalized for keyword in keywords) and bool(
        re.search(r"\d+", normalized)
    )


def _copy_excel_range_as_png(
    w3c,
    xls_ws,
    row_from: int,
    col_from: int,
    row_to: int,
    col_to: int,
    output_path: Path,
    image_grab,
    clipboard_retries: int,
    clipboard_sleep_seconds: float,
) -> None:
    """
    Copy an Excel range as a bitmap and save clipboard image to PNG.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    xls_range = xls_ws.Range(
        xls_ws.Cells(row_from, col_from),
        xls_ws.Cells(row_to, col_to),
    )

    last_error: Optional[str] = None

    for attempt in range(1, int(clipboard_retries) + 1):
        try:
            xls_range.CopyPicture(Format=EXCEL_XL_BITMAP)
            time.sleep(float(clipboard_sleep_seconds))
            image = image_grab.grabclipboard()

            if image is not None:
                image.save(str(output_path))
                return

            last_error = "Clipboard did not contain an image."

        except Exception as exc:
            last_error = str(exc)

        time.sleep(float(clipboard_sleep_seconds))

    raise RuntimeError(
        f"Failed to copy Excel range {row_from}:{col_from} to {row_to}:{col_to} "
        f"as PNG after {clipboard_retries} attempt(s): {last_error}"
    )