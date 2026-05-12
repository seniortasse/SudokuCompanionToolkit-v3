from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from python.publishing.cleanup import (
    DeleteAction,
    analyze_candidate_jsonl_delete,
    confirmation_token_for_plan,
    decide_candidate_jsonl_delete,
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
from python.publishing.puzzle_catalog.candidate_jsonl_admin import (
    archive_candidate_jsonl_lines,
    load_candidate_jsonl_lines,
    rewrite_candidate_jsonl_without_selected,
    select_candidate_jsonl_lines,
    summarize_selected_candidate_lines,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete raw candidates from candidates.jsonl safely."
    )
    parser.add_argument(
        "--jsonl",
        default="runs/publishing/classic9/puzzle_generation/candidates.jsonl",
    )
    parser.add_argument("--line-number", action="append", type=int, dest="line_numbers")
    parser.add_argument("--generation-seed", action="append", type=int, dest="generation_seeds")
    parser.add_argument("--pattern-id", action="append", dest="pattern_ids")
    parser.add_argument("--givens81", action="append", dest="givens81_values")
    parser.add_argument("--solution81", action="append", dest="solution81_values")
    parser.add_argument(
        "--records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
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
        "--archive-removed-to",
        default=None,
        help="Optional JSONL file that will receive removed raw lines.",
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


def run_delete_candidates_jsonl(
    *,
    jsonl_path: Path,
    records_dir: Path,
    backup_root: Path,
    report_dir: Path,
    archive_removed_to: Path | None = None,
    line_numbers: Sequence[int] | None = None,
    generation_seeds: Sequence[int] | None = None,
    pattern_ids: Sequence[str] | None = None,
    givens81_values: Sequence[str] | None = None,
    solution81_values: Sequence[str] | None = None,
    dry_run: bool = False,
    yes: bool = False,
    confirm: str | None = None,
) -> int:
    plan = analyze_candidate_jsonl_delete(
        jsonl_path=jsonl_path,
        requested_action=DeleteAction.RAW_DELETE,
        line_numbers=list(line_numbers or []),
        generation_seeds=list(generation_seeds or []),
        pattern_ids=list(pattern_ids or []),
        givens81_values=list(givens81_values or []),
        solution81_values=list(solution81_values or []),
        records_dir=records_dir,
    )
    decision = decide_candidate_jsonl_delete(plan)

    print(render_delete_plan(plan), flush=True)
    print("", flush=True)
    print(render_policy_decision(decision), flush=True)

    preview_report = write_delete_report(
        plan=plan,
        decision=decision,
        report_dir=report_dir,
        stage="preview",
        extra={
            "workflow": "delete_candidates_jsonl",
            "jsonl_path": str(jsonl_path),
            "selector": {
                "line_numbers": list(line_numbers or []),
                "generation_seeds": list(generation_seeds or []),
                "pattern_ids": list(pattern_ids or []),
                "givens81_values": list(givens81_values or []),
                "solution81_values": list(solution81_values or []),
            },
        },
    )
    print("", flush=True)
    print(render_report_summary(preview_report, stage="preview"), flush=True)

    if decision.outcome == "BLOCK":
        print("", flush=True)
        print("Delete is blocked. No mutation was performed.", flush=True)
        return 1

    if dry_run:
        print("", flush=True)
        print("Dry run requested. No mutation was performed.", flush=True)
        return 0

    confirmed = is_confirmation_satisfied(
        plan=plan,
        provided_token=confirm,
        yes=yes,
        is_batch=False,
        is_delete_all=False,
    )
    if not confirmed:
        spec = confirmation_token_for_plan(
            plan,
            is_batch=False,
            is_delete_all=False,
        )
        print("", flush=True)
        print(
            f"Confirmation required before mutation. Expected token: {spec.required_token}",
            flush=True,
        )
        confirmed = prompt_for_confirmation(
            plan=plan,
            is_batch=False,
            is_delete_all=False,
        )

    if not confirmed:
        print("Confirmation failed. No mutation was performed.", flush=True)
        return 1

    snapshot_dir = write_backup_snapshot(
        plan=plan,
        backup_root=backup_root,
    )
    print("", flush=True)
    print(render_snapshot_summary(snapshot_dir), flush=True)

    try:
        entries = load_candidate_jsonl_lines(jsonl_path)
        selected_entries = select_candidate_jsonl_lines(
            entries,
            line_numbers=line_numbers,
            generation_seeds=generation_seeds,
            pattern_ids=pattern_ids,
            givens81_values=givens81_values,
            solution81_values=solution81_values,
        )
        selection_summary = summarize_selected_candidate_lines(selected_entries)

        archive_path = None
        if archive_removed_to is not None and selected_entries:
            archive_path = archive_candidate_jsonl_lines(
                selected_entries=selected_entries,
                archive_path=archive_removed_to,
            )

        removed_count, kept_count = rewrite_candidate_jsonl_without_selected(
            jsonl_path=jsonl_path,
            selected_entries=selected_entries,
        )

        execute_report = write_delete_report(
            plan=plan,
            decision=decision,
            report_dir=report_dir,
            stage="execute",
            snapshot_dir=snapshot_dir,
            extra={
                "workflow": "delete_candidates_jsonl",
                "removed_count": removed_count,
                "kept_count": kept_count,
                "archive_removed_to": str(archive_path) if archive_path else None,
                "selection_summary": selection_summary,
            },
        )
        print(
            f"Removed {removed_count} candidate line(s) from {jsonl_path}; kept {kept_count} line(s).",
            flush=True,
        )
        if archive_path:
            print(f"Archived removed lines to: {archive_path}", flush=True)
        print(render_report_summary(execute_report, stage="execute"), flush=True)
        return 0

    except Exception as exc:
        restored_paths = restore_backup_snapshot(snapshot_dir=snapshot_dir)
        rollback_report = write_delete_report(
            plan=plan,
            decision=decision,
            report_dir=report_dir,
            stage="rollback",
            snapshot_dir=snapshot_dir,
            extra={
                "workflow": "delete_candidates_jsonl",
                "error": str(exc),
                "restored_paths": restored_paths,
            },
        )
        print(f"Mutation failed: {exc}", flush=True)
        print(render_report_summary(rollback_report, stage="rollback"), flush=True)
        return 1


def main() -> int:
    args = _parse_args()
    archive_removed_to = Path(args.archive_removed_to) if args.archive_removed_to else None

    return run_delete_candidates_jsonl(
        jsonl_path=Path(args.jsonl),
        records_dir=Path(args.records_dir),
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        archive_removed_to=archive_removed_to,
        line_numbers=args.line_numbers,
        generation_seeds=args.generation_seeds,
        pattern_ids=args.pattern_ids,
        givens81_values=args.givens81_values,
        solution81_values=args.solution81_values,
        dry_run=bool(args.dry_run),
        yes=bool(args.yes),
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())