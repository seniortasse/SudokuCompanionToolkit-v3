from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from python.publishing.step_solutions.locale_templates import (
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.runtime_qa import (
    DEFAULT_BOOKS_ROOT,
    DEFAULT_STEP_SOLUTION_PACKAGES_ROOT,
)
from python.publishing.step_solutions.progress import print_progress


DEFAULT_PACKAGE_WITH_QA_REPORT = (
    Path("runs/publishing/classic9/step_solution_package_with_qa")
    / "package_with_qa_report.json"
)


def _split_values(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export localized step-solution packages and immediately run runtime QA."
        )
    )
    parser.add_argument(
        "--book-id",
        required=True,
        help="Book id, for example BK-CL9-DW-B01.",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        required=True,
        help="Locales to export. Supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_STEP_SOLUTION_PACKAGES_ROOT,
        help="Root folder for generated step-solution packages.",
    )
    parser.add_argument(
        "--books-root",
        type=Path,
        default=DEFAULT_BOOKS_ROOT,
        help="Root folder containing book folders.",
    )
    parser.add_argument(
        "--only-puzzle",
        default=None,
        help="Optional internal puzzle code to export, for example L1-001.",
    )
    parser.add_argument(
        "--only-section",
        default=None,
        help="Optional section code/id filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum puzzle count.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing assets.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing assets when supported.",
    )
    parser.add_argument(
        "--excel-visible",
        action="store_true",
        help="Show Excel during image export.",
    )
    parser.add_argument(
        "--max-workbooks-to-check",
        type=int,
        default=3,
        help="Maximum workbooks to inspect per locale during runtime QA.",
    )
    parser.add_argument(
        "--max-csv-rows-to-check",
        type=int,
        default=25,
        help="Maximum CSV rows to inspect per locale during runtime QA.",
    )
    parser.add_argument(
        "--allow-english-leakage",
        action="store_true",
        help="Downgrade detected English leakage from error to warning during QA.",
    )
    parser.add_argument(
        "--skip-runtime-qa",
        action="store_true",
        help="Only export packages; do not run runtime QA.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_PACKAGE_WITH_QA_REPORT,
        help="Final orchestration report JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    report: Dict[str, Any] = {
        "phase": "phase_l7_export_step_solution_packages_with_qa",
        "book_id": args.book_id,
        "locales": locales,
        "selection": {
            "only_puzzle": args.only_puzzle,
            "only_section": args.only_section,
            "limit": args.limit,
        },
        "options": {
            "force": args.force,
            "skip_existing": args.skip_existing,
            "excel_visible": args.excel_visible,
            "skip_runtime_qa": args.skip_runtime_qa,
            "allow_english_leakage": args.allow_english_leakage,
            "max_workbooks_to_check": args.max_workbooks_to_check,
            "max_csv_rows_to_check": args.max_csv_rows_to_check,
        },
        "steps": [],
        "summary": {
            "ok": True,
            "failed_step_count": 0,
        },
    }

    export_command = [
        sys.executable,
        "-m",
        "python.publishing.workflows.export_step_solution_package",
        "--book-id",
        args.book_id,
        "--locales",
        *locales,
        "--output-root",
        str(args.output_root),
        "--books-root",
        str(args.books_root),
    ]

    if args.only_puzzle:
        export_command.extend(["--only-puzzle", args.only_puzzle])
    if args.only_section:
        export_command.extend(["--only-section", args.only_section])
    if args.limit is not None:
        export_command.extend(["--limit", str(args.limit)])
    if args.force:
        export_command.append("--force")
    if args.skip_existing:
        export_command.append("--skip-existing")
    if args.excel_visible:
        export_command.append("--excel-visible")

    _run_step(
        report=report,
        name="export_step_solution_package",
        command=export_command,
    )

    if args.skip_runtime_qa:
        _skip_step(report, "runtime_qa")
    else:
        qa_command = [
            sys.executable,
            "-m",
            "python.publishing.workflows.qa_step_solution_package_runtime",
            "--book-id",
            args.book_id,
            "--locales",
            *locales,
            "--output-root",
            str(args.output_root),
            "--books-root",
            str(args.books_root),
            "--max-workbooks-to-check",
            str(args.max_workbooks_to_check),
            "--max-csv-rows-to-check",
            str(args.max_csv_rows_to_check),
        ]

        if args.allow_english_leakage:
            qa_command.append("--allow-english-leakage")

        _run_step(
            report=report,
            name="runtime_qa",
            command=qa_command,
        )

    failed_steps = [
        step for step in report["steps"]
        if step.get("status") == "failed"
    ]
    report["summary"]["failed_step_count"] = len(failed_steps)
    report["summary"]["ok"] = len(failed_steps) == 0

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_json"] = str(args.report_json)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if report["summary"]["ok"] else 1


def _skip_step(report: Dict[str, Any], name: str) -> None:
    report["steps"].append(
        {
            "name": name,
            "status": "skipped",
            "command": [],
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    )


def _run_step(
    report: Dict[str, Any],
    name: str,
    command: List[str],
) -> None:
    """
    Run one child workflow while streaming stdout/stderr live.

    This keeps the final report behavior from the old implementation:
        - command
        - returncode
        - stdout_tail
        - stderr_tail

    But unlike subprocess.run(capture_output=True), it also prints child output
    as it happens, so long exports show live progress in PowerShell.
    """

    print_progress(
        "WRAPPER",
        f"{name} started | command={_command_for_display(command)}",
    )

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, sys.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, sys.stderr, stderr_chunks),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()

    stdout_thread.join()
    stderr_thread.join()

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    status = "ok" if returncode == 0 else "failed"

    print_progress(
        "WRAPPER",
        f"{name} {status.upper()} | returncode={returncode}",
    )

    step = {
        "name": name,
        "status": status,
        "command": command,
        "returncode": returncode,
        "stdout_tail": _tail(stdout_text),
        "stderr_tail": _tail(stderr_text),
    }
    report["steps"].append(step)


def _stream_pipe(
    pipe: Optional[TextIO],
    output_stream: TextIO,
    chunks: List[str],
) -> None:
    """
    Read a subprocess pipe line by line, print it live, and keep a copy.

    The copy is later trimmed into stdout_tail/stderr_tail for the JSON report.
    """

    if pipe is None:
        return

    try:
        for line in pipe:
            chunks.append(line)
            print(line, end="", file=output_stream, flush=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _command_for_display(command: List[str]) -> str:
    """
    Compact command string for the live wrapper progress line.
    """

    return " ".join(str(part) for part in command)

def _tail(text: Optional[str], max_chars: int = 12000) -> str:
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())