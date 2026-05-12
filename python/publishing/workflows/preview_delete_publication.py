from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.cleanup import (
    DeleteAction,
    analyze_publication_delete,
    analyze_publication_spec_delete,
    decide_publication_delete,
    decide_publication_spec_delete,
    render_delete_plan,
    render_policy_decision,
    render_report_summary,
    write_delete_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview delete safety for publication directories and publication spec files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--publication-id",
        action="append",
        dest="publication_ids",
        help="Repeat --publication-id to preview multiple publication directories.",
    )
    group.add_argument(
        "--publication-spec-id",
        action="append",
        dest="publication_spec_ids",
        help="Repeat --publication-spec-id to preview multiple publication spec targets.",
    )

    parser.add_argument(
        "--publications-dir",
        default="datasets/sudoku_books/classic9/publications",
    )
    parser.add_argument(
        "--publication-specs-dir",
        default="datasets/sudoku_books/classic9/publication_specs",
    )
    parser.add_argument(
        "--cascade-publication-specs",
        action="store_true",
        help="Preview as if linked publication spec files would also be removed.",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    blocked_count = 0

    if args.publication_ids:
        for index, publication_id in enumerate(args.publication_ids, start=1):
            plan = analyze_publication_delete(
                publication_id=publication_id,
                requested_action=DeleteAction.DELETE,
                publications_dir=Path(args.publications_dir),
                publication_specs_dir=Path(args.publication_specs_dir),
            )

            if args.cascade_publication_specs:
                for dep in plan.dependencies:
                    if dep.dependency_type == "publication_spec" and dep.path not in plan.files_to_delete:
                        plan.files_to_delete.append(dep.path)

            decision = decide_publication_delete(
                plan,
                cascade_publication_specs=bool(args.cascade_publication_specs),
            )

            if index > 1:
                print("", flush=True)

            print(render_delete_plan(plan), flush=True)
            print("", flush=True)
            print(render_policy_decision(decision), flush=True)

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=Path(args.report_dir),
                stage="preview",
                extra={
                    "workflow": "preview_delete_publication",
                    "publication_id": publication_id,
                    "cascade_publication_specs": bool(args.cascade_publication_specs),
                },
            )
            print("", flush=True)
            print(render_report_summary(report_path, stage="preview"), flush=True)

            if decision.outcome == "BLOCK":
                blocked_count += 1

    if args.publication_spec_ids:
        for index, publication_spec_id in enumerate(args.publication_spec_ids, start=1):
            plan = analyze_publication_spec_delete(
                publication_spec_id=publication_spec_id,
                requested_action=DeleteAction.DELETE,
                publication_specs_dir=Path(args.publication_specs_dir),
            )
            decision = decide_publication_spec_delete(plan)

            if index > 1:
                print("", flush=True)

            print(render_delete_plan(plan), flush=True)
            print("", flush=True)
            print(render_policy_decision(decision), flush=True)

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=Path(args.report_dir),
                stage="preview",
                extra={
                    "workflow": "preview_delete_publication",
                    "publication_spec_id": publication_spec_id,
                },
            )
            print("", flush=True)
            print(render_report_summary(report_path, stage="preview"), flush=True)

            if decision.outcome == "BLOCK":
                blocked_count += 1

    return 1 if blocked_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())