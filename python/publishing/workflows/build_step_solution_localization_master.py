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
    DEFAULT_LOCALIZATION_MASTER_IMPORT_REPORT,
    DEFAULT_LOCALIZATION_MASTER_XLSX,
    build_step_solution_localization_master,
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
            "Build the step-solution localization master workbook from the "
            "English Template_Messages.xlsx schema and current localized seed files."
        )
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["en", "fr", "de", "it", "es"],
        help="Locales to import. English is always included as schema master.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Root solution_templates folder.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=DEFAULT_LOCALIZATION_MASTER_XLSX,
        help="Output localization master workbook.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_LOCALIZATION_MASTER_IMPORT_REPORT,
        help="Output import report JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing localization master workbook.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    if "en" not in locales:
        locales = ["en"] + locales

    report = build_step_solution_localization_master(
        templates_root=args.templates_root,
        locales=locales,
        output_xlsx=args.output_xlsx,
        report_json=args.report_json,
        overwrite=args.overwrite,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())