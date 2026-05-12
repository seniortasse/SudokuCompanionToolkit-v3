from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.template_auditor import (
    audit_step_solution_templates,
    write_template_audit_report,
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
            "Audit localized step-solution Template_Messages.xlsx files against "
            "the English schema master. This workflow does not modify templates."
        )
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["fr", "de", "it", "es"],
        help="Locales to audit. Supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Root solution_templates folder.",
    )
    parser.add_argument(
        "--english-template",
        type=Path,
        default=None,
        help="Optional explicit path to English Template_Messages.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/publishing/classic9/step_solution_template_audit"),
        help="Output folder for audit reports.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as non-OK in the final report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    locales = [
        normalize_step_solution_locale(locale)
        for locale in _split_values(args.locales)
    ]

    report = audit_step_solution_templates(
        locales=locales,
        templates_root=args.templates_root,
        english_template_path=args.english_template,
        strict=args.strict,
    )

    paths = write_template_audit_report(
        report=report,
        output_dir=args.output_dir,
    )

    payload = report.to_dict()
    payload["report_paths"] = {key: str(value) for key, value in paths.items()}

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())