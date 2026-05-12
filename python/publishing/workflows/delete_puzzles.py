from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from python.publishing.cleanup import (
    DeleteAction,
    analyze_record_delete,
    confirmation_token_for_plan,
    decide_record_delete,
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
from python.publishing.puzzle_catalog.catalog_admin import (
    archive_candidate_record,
    delete_candidate_record,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive or delete canonical puzzle records safely."
    )
    parser.add_argument(
        "--record-id",
        required=True,
        action="append",
        dest="record_ids",
        help="Repeat --record-id to process multiple records.",
    )
    parser.add_argument(
        "--action",
        choices=[DeleteAction.ARCHIVE, DeleteAction.DELETE],
        default=DeleteAction.DELETE,
    )
    parser.add_argument("--library-id", required=True)
    parser.add_argument(
        "--records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
    )
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
    )
    parser.add_argument(
        "--books-dir",
        default="datasets/sudoku_books/classic9/books",
    )
    parser.add_argument(
        "--publications-dir",
        default="datasets/sudoku_books/classic9/publications",
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


def run_delete_puzzles(
    *,
    record_ids: Sequence[str],
    action: str,
    library_id: str,
    records_dir: Path,
    inventory_dir: Path,
    books_dir: Path,
    publications_dir: Path,
    backup_root: Path,
    report_dir: Path,
    dry_run: bool = False,
    yes: bool = False,
    confirm: str | None = None,
) -> int:
    plans = []
    decisions = []
    blocked_count = 0

    for index, record_id in enumerate(record_ids, start=1):
        plan = analyze_record_delete(
            record_id=record_id,
            library_id=library_id,
            requested_action=action,
            records_dir=records_dir,
            inventory_dir=inventory_dir,
            books_dir=books_dir,
            publications_dir=publications_dir,
        )
        decision = decide_record_delete(plan)

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
                "workflow": "delete_puzzles",
                "record_id": record_id,
            },
        )
        print("", flush=True)
        print(render_report_summary(report_path, stage="preview"), flush=True)

        if decision.outcome == "BLOCK":
            blocked_count += 1

    if blocked_count > 0:
        print("", flush=True)
        print(f"Blocked records: {blocked_count}. No mutation was performed.", flush=True)
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
            if decision.action == DeleteAction.ARCHIVE:
                ok, message = archive_candidate_record(
                    records_dir=records_dir,
                    record_id=plan.target.target_id,
                )
            elif decision.action == DeleteAction.DELETE:
                ok, message = delete_candidate_record(
                    records_dir=records_dir,
                    record_id=plan.target.target_id,
                )
            else:
                raise RuntimeError(
                    f"Unsupported resolved action for puzzle workflow: {decision.action}"
                )

            if not ok:
                raise RuntimeError(message)

            report_path = write_delete_report(
                plan=plan,
                decision=decision,
                report_dir=report_dir,
                stage="execute",
                snapshot_dir=snapshot_dir,
                extra={
                    "workflow": "delete_puzzles",
                    "message": message,
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
                    "workflow": "delete_puzzles",
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
    return run_delete_puzzles(
        record_ids=args.record_ids,
        action=str(args.action),
        library_id=args.library_id,
        records_dir=Path(args.records_dir),
        inventory_dir=Path(args.inventory_dir),
        books_dir=Path(args.books_dir),
        publications_dir=Path(args.publications_dir),
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        dry_run=bool(args.dry_run),
        yes=bool(args.yes),
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())