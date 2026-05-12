from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.localization_master import (
    DEFAULT_LOCALIZATION_MASTER_XLSX,
)
from python.publishing.step_solutions.apply_localization_seed import (
    DEFAULT_APPLY_SEED_REPORT,
)
from python.publishing.step_solutions.template_generator import (
    DEFAULT_GENERATED_TEMPLATE_REPORT,
)


DEFAULT_PREPARE_LOCALIZATION_REPORT = (
    Path("runs/publishing/classic9/step_solution_localization_prepare")
    / "prepare_localization_templates_report.json"
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
            "Prepare localized step-solution Template_Messages.xlsx files. "
            "This orchestrates: build master, apply seed, generate templates, audit."
        )
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["fr", "de", "it", "es"],
        help=(
            "Locales to generate/audit. Supports space-separated or comma-separated values. "
            "English is automatically included while building the master."
        ),
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Root solution_templates folder.",
    )
    parser.add_argument(
        "--master-xlsx",
        type=Path,
        default=DEFAULT_LOCALIZATION_MASTER_XLSX,
        help="Localization master workbook path.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_PREPARE_LOCALIZATION_REPORT,
        help="Final orchestration report JSON.",
    )
    parser.add_argument(
        "--skip-build-master",
        action="store_true",
        help="Skip rebuilding localization master workbook.",
    )
    parser.add_argument(
        "--skip-apply-seed",
        action="store_true",
        help="Skip applying curated seed translations to the localization master.",
    )
    parser.add_argument(
        "--skip-generate-templates",
        action="store_true",
        help="Skip generating localized Template_Messages.xlsx files.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip template audit.",
    )
    parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="Run template audit in strict mode.",
    )
    parser.add_argument(
        "--overwrite-master",
        action="store_true",
        default=True,
        help="Overwrite existing localization master when building it. Default: true.",
    )
    parser.add_argument(
        "--no-overwrite-master",
        dest="overwrite_master",
        action="store_false",
        help="Do not overwrite existing localization master.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backups when applying seed/generating templates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    master_locales = ["en"] + [locale for locale in locales if locale != "en"]

    report: Dict[str, Any] = {
        "phase": "phase_l7_prepare_step_solution_localization_templates",
        "locales": locales,
        "master_locales": master_locales,
        "templates_root": str(args.templates_root),
        "master_xlsx": str(args.master_xlsx),
        "steps": [],
        "summary": {
            "ok": True,
            "failed_step_count": 0,
        },
    }

    if not args.skip_build_master:
        _run_step(
            report=report,
            name="build_localization_master",
            command=[
                sys.executable,
                "-m",
                "python.publishing.workflows.build_step_solution_localization_master",
                "--locales",
                *master_locales,
                "--templates-root",
                str(args.templates_root),
                "--output-xlsx",
                str(args.master_xlsx),
                "--report-json",
                str(args.master_xlsx.with_suffix(".import_report.json")),
                *(
                    ["--overwrite"]
                    if args.overwrite_master
                    else []
                ),
            ],
        )
    else:
        _skip_step(report, "build_localization_master")

    if not args.skip_apply_seed:
        _run_step(
            report=report,
            name="apply_localization_seed",
            command=[
                sys.executable,
                "-m",
                "python.publishing.workflows.apply_step_solution_localization_seed",
                "--master-xlsx",
                str(args.master_xlsx),
                "--report-json",
                str(DEFAULT_APPLY_SEED_REPORT),
                "--locales",
                *[locale for locale in locales if locale != "en"],
                *(
                    ["--no-backup"]
                    if args.no_backup
                    else []
                ),
            ],
        )
    else:
        _skip_step(report, "apply_localization_seed")

    if not args.skip_generate_templates:
        _run_step(
            report=report,
            name="generate_localized_templates",
            command=[
                sys.executable,
                "-m",
                "python.publishing.workflows.generate_step_solution_templates",
                "--master-xlsx",
                str(args.master_xlsx),
                "--templates-root",
                str(args.templates_root),
                "--locales",
                *[locale for locale in locales if locale != "en"],
                "--report-json",
                str(DEFAULT_GENERATED_TEMPLATE_REPORT),
                *(
                    ["--no-backup"]
                    if args.no_backup
                    else []
                ),
            ],
        )
    else:
        _skip_step(report, "generate_localized_templates")

    if not args.skip_audit:
        _run_step(
            report=report,
            name="audit_localized_templates",
            command=[
                sys.executable,
                "-m",
                "python.publishing.workflows.audit_step_solution_templates",
                "--templates-root",
                str(args.templates_root),
                "--locales",
                *[locale for locale in locales if locale != "en"],
                *(
                    ["--strict"]
                    if args.strict_audit
                    else []
                ),
            ],
        )
    else:
        _skip_step(report, "audit_localized_templates")

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
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )

    step = {
        "name": name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }
    report["steps"].append(step)


def _tail(text: Optional[str], max_chars: int = 8000) -> str:
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())