from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    build_template_status_report,
    copy_legacy_templates_to_canonical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare canonical step-solution templates by copying the active "
            "legacy Excel templates into datasets/sudoku_books/classic9/"
            "solution_templates."
        )
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("python/step-by-step_solutions"),
        help="Legacy step-by-step_solutions folder.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=DEFAULT_SOLUTION_TEMPLATES_ROOT,
        help="Canonical solution_templates folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canonical template files.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only print current canonical template status; do not copy files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.status_only:
        report = build_template_status_report(
            templates_root=args.templates_root,
        )
    else:
        report = copy_legacy_templates_to_canonical(
            legacy_root=args.legacy_root,
            templates_root=args.templates_root,
            overwrite=args.overwrite,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())