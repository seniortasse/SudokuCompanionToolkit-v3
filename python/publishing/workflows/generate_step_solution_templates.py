from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.localization_master import (
    DEFAULT_LOCALIZATION_MASTER_XLSX,
)
from python.publishing.step_solutions.template_generator import (
    DEFAULT_GENERATED_TEMPLATE_REPORT,
    generate_step_solution_templates_from_master,
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
            "Generate localized Template_Messages.xlsx files from the "
            "step-solution localization master workbook."
        )
    )
    parser.add_argument(
        "--master-xlsx",
        type=Path,
        default=DEFAULT_LOCALIZATION_MASTER_XLSX,
        help="Localization master workbook.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Root solution_templates folder.",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["fr", "de", "it", "es"],
        help="Locales to generate. Supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--english-template",
        type=Path,
        default=None,
        help="Optional explicit English Template_Messages.xlsx schema source.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_GENERATED_TEMPLATE_REPORT,
        help="Generation report JSON.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up existing localized Template_Messages.xlsx files.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if target localized Template_Messages.xlsx already exists.",
    )
    parser.add_argument(
        "--allow-placeholder-mismatch",
        action="store_true",
        help="Allow placeholder mismatches. Not recommended.",
    )
    parser.add_argument(
        "--generate-english-copy",
        action="store_true",
        help="Generate Template_Messages.generated.xlsx for English as a dry comparison artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    report = generate_step_solution_templates_from_master(
        master_xlsx=args.master_xlsx,
        templates_root=args.templates_root,
        locales=locales,
        english_template_path=args.english_template,
        backup=not args.no_backup,
        overwrite=not args.no_overwrite,
        report_json=args.report_json,
        fail_on_placeholder_mismatch=not args.allow_placeholder_mismatch,
        generate_english_copy=args.generate_english_copy,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())