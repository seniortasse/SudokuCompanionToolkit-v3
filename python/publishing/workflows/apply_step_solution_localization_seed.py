from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from python.publishing.step_solutions.apply_localization_seed import (
    DEFAULT_APPLY_SEED_REPORT,
    apply_localization_seed_to_master,
)
from python.publishing.step_solutions.localization_master import (
    DEFAULT_LOCALIZATION_MASTER_XLSX,
)
from python.publishing.step_solutions.locale_templates import normalize_step_solution_locale


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
            "Apply curated French/German/Italian/Spanish translation seeds to "
            "the step-solution localization master workbook."
        )
    )
    parser.add_argument(
        "--master-xlsx",
        type=Path,
        default=DEFAULT_LOCALIZATION_MASTER_XLSX,
        help="Localization master workbook to update.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_APPLY_SEED_REPORT,
        help="Seed-application report JSON.",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["fr", "de", "it", "es"],
        help="Locales to update. Supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup copy of the master workbook before editing.",
    )
    parser.add_argument(
        "--allow-placeholder-mismatch",
        action="store_true",
        help="Do not fail on placeholder mismatches. Not recommended.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    report = apply_localization_seed_to_master(
        master_xlsx=args.master_xlsx,
        report_json=args.report_json,
        locales=locales,
        backup=not args.no_backup,
        fail_on_placeholder_mismatch=not args.allow_placeholder_mismatch,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())