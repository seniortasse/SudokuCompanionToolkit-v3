from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Sequence

from python.publishing.cleanup import (
    DeleteAction,
    analyze_publication_delete,
    analyze_publication_spec_delete,
    confirmation_token_for_plan,
    decide_publication_delete,
    decide_publication_spec_delete,
    is_confirmation_satisfied,
    prompt_for_confirmation,
    render_delete_plan,
    render_policy_decision,
    render_report_summary,
    render_snapshot_summary,
    restore_backup_snapshot,
    write_backup_snapshot,
    write_delete_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete publication directories or publication spec files safely."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--publication-id",
        action="append",
        dest="publication_ids",
        help="Repeat --publication-id to delete multiple publication directories.",
    )
    group.add_argument(
        "--publication-spec-id",
        action="append",
        dest="publication_spec_ids",
        help="Repeat --publication-spec-id to delete multiple publication spec targets.",
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
        "--backup-root",
        default="runs/publishing/backups",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    parser.add_argument(
        "--cascade-publication-specs",
        action="store_true",
        help="When deleting a publication directory, also remove linked publication spec files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze, preview, and report without mutating anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass interactive confirmation.",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        help="Provide the required confirmation token explicitly.",
    )
    return parser.parse_args()


def _safe_remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _apply_publication_delete_mutation(
    *,
    plan,
    delete_spec_paths: Sequence[Path],
) -> str:
    removed_paths: List[Path] = []

    publication_dir = Path(plan.metadata["publication_dir"])
    _safe_remove_path(publication_dir)
    removed_paths.append(publication_dir)

    for spec_path in delete_spec_paths:
        _safe_remove_path(spec_path)
        removed_paths.append(spec_path)

    removed_count = len([p for p in removed_paths if not p.exists()])
    return (
        f"Deleted publication {plan.target.target_id}; removed {removed_count} path(s)."
    )


def _apply_publication_spec_delete_mutation(
    *,
    plan,
) -> str:
    removed_paths: List[Path] = []
    for path_str in plan.files_to_delete:
        path = Path(path_str)
        _safe_remove_path(path)
        removed_paths.append(path)

    removed_count = len([p for p in removed_paths if not p.exists()])
    return (
        f"Deleted publication spec target {plan.target.target_id}; removed {removed_count} path(s)."
    )


def run_delete_publications(
    *,
    publication_ids: Sequence[str] | None = None,
    publication_spec_ids: Sequence[str] | None = None,
    publications_dir: Path,
    publication_specs_dir: Path,
    backup_root: Path,
    report_dir: Path,
    cascade_publication_specs: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    confirm: str | None = None,
) -> int:
    plans = []
    decisions = []
    blocked_count = 0

    if publication_ids:
        for index, publication_id in enumerate(publication_ids, start=1):
            plan = analyze_publication_delete(
                publication_id=publication_id,
                requested_action=DeleteAction.DELETE,
                publications_dir=publications_dir,
                publication_specs_dir=publication_specs_dir,
            )

            if cascade_publication_specs:
                for dep in plan.dependencies:
                    if dep.dependency_type == "publication_spec" and dep.path not in plan.files_to_delete:
                        plan.files_to_delete.append(dep.path)

            decision = decide_publication_delete(
                plan,
                cascade_publication_specs=bool(cascade_publication_specs),
            )

            plans.append(plan)
            decisions.append(decision)

            if index > 1:
                print("", flush=True)

            print(render_delete_plan(plan), flush=True)
            print("", flush=True)
            print(render_policy_decision(decision), flush=True)

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="preview",
                extra={
                    "workflow": "delete_publications",
                    "publication_id": publication_id,
                    "cascade_publication_specs": bool(cascade_publication_specs),
                },
            )
            print("", flush=True)
            print(render_report_summary(report_path, stage="preview"), flush=True)

            if decision.outcome == "BLOCK":
                blocked_count += 1

    if publication_spec_ids:
        for index, publication_spec_id in enumerate(publication_spec_ids, start=1):
            plan = analyze_publication_spec_delete(
                publication_spec_id=publication_spec_id,
                requested_action=DeleteAction.DELETE,
                publication_specs_dir=publication_specs_dir,
            )
            decision = decide_publication_spec_delete(plan)

            plans.append(plan)
            decisions.append(decision)

            if index > 1:
                print("", flush=True)

            print(render_delete_plan(plan), flush=True)
            print("", flush=True)
            print(render_policy_decision(decision), flush=True)

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="preview",
                extra={
                    "workflow": "delete_publications",
                    "publication_spec_id": publication_spec_id,
                },
            )
            print("", flush=True)
            print(render_report_summary(report_path, stage="preview"), flush=True)

            if decision.outcome == "BLOCK":
                blocked_count += 1

    if blocked_count > 0:
        print("", flush=True)
        print(f"Blocked targets: {blocked_count}. No mutation was performed.", flush=True)
        return 1

    if dry_run:
        print("", flush=True)
        print("Dry run requested. No mutation was performed.", flush=True)
        return 0

    first_plan = plans[0]
    is_batch = len(plans) > 1

    confirmed = is_confirmation_satisfied(
        plan=first_plan,
        provided_token=confirm,
        yes=yes,
        is_batch=is_batch,
        is_delete_all=False,
    )
    if not confirmed:
        spec = confirmation_token_for_plan(
            first_plan,
            is_batch=is_batch,
            is_delete_all=False,
        )
        print("", flush=True)
        print(
            f"Confirmation required before mutation. Expected token: {spec.required_token}",
            flush=True,
        )
        confirmed = prompt_for_confirmation(
            plan=first_plan,
            is_batch=is_batch,
            is_delete_all=False,
        )

    if not confirmed:
        print("Confirmation failed. No mutation was performed.", flush=True)
        return 1

    for plan, decision in zip(plans, decisions):
        snapshot_dir = write_backup_snapshot(
            plan=plan,
            backup_root=backup_root,
        )
        print("", flush=True)
        print(render_snapshot_summary(snapshot_dir), flush=True)

        try:
            if plan.target.target_type == "publication":
                spec_paths: List[Path] = []
                if decision.action == DeleteAction.CASCADE_DELETE:
                    for dep in plan.dependencies:
                        if dep.dependency_type == "publication_spec":
                            spec_paths.append(Path(dep.path))
                message = _apply_publication_delete_mutation(
                    plan=plan,
                    delete_spec_paths=spec_paths,
                )
            elif plan.target.target_type == "publication_spec":
                message = _apply_publication_spec_delete_mutation(
                    plan=plan,
                )
            else:
                raise RuntimeError(
                    f"Unsupported target_type for delete_publications workflow: {plan.target.target_type}"
                )

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="execute",
                snapshot_dir=snapshot_dir,
                extra={
                    "workflow": "delete_publications",
                    "message": message,
                    "cascade_publication_specs": bool(cascade_publication_specs),
                },
            )
            print(message, flush=True)
            print(render_report_summary(report_path, stage="execute"), flush=True)

        except Exception as exc:
            restored_paths = restore_backup_snapshot(snapshot_dir=snapshot_dir)
            rollback_report = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="rollback",
                snapshot_dir=snapshot_dir,
                extra={
                    "workflow": "delete_publications",
                    "error": str(exc),
                    "restored_paths": restored_paths,
                },
            )
            print(f"Mutation failed for {plan.target.target_id}: {exc}", flush=True)
            print(render_report_summary(rollback_report, stage="rollback"), flush=True)
            return 1

    return 0


def main() -> int:
    args = _parse_args()
    return run_delete_publications(
        publication_ids=args.publication_ids,
        publication_spec_ids=args.publication_spec_ids,
        publications_dir=Path(args.publications_dir),
        publication_specs_dir=Path(args.publication_specs_dir),
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        cascade_publication_specs=bool(args.cascade_publication_specs),
        dry_run=bool(args.dry_run),
        yes=bool(args.yes),
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())