from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from python.publishing.cleanup import (
    DeleteAction,
    analyze_candidate_jsonl_delete,
    confirmation_token_for_plan,
    is_confirmation_satisfied,
    prompt_for_confirmation,
)
from python.publishing.puzzle_catalog.candidate_jsonl_admin import load_candidate_jsonl_lines
from python.publishing.workflows.delete_candidates_jsonl import run_delete_candidates_jsonl


def _select_generation_pool_targets(
    *,
    jsonl_path: Path,
    all_lines: bool,
    line_numbers: List[int] | None,
    generation_seeds: List[int] | None,
    pattern_ids: List[str] | None,
    givens81_values: List[str] | None,
    solution81_values: List[str] | None,
) -> dict:
    if all_lines:
        entries = load_candidate_jsonl_lines(jsonl_path)
        return {
            "line_numbers": [int(entry["_line_number"]) for entry in entries],
            "generation_seeds": [],
            "pattern_ids": [],
            "givens81_values": [],
            "solution81_values": [],
        }

    return {
        "line_numbers": list(line_numbers or []),
        "generation_seeds": list(generation_seeds or []),
        "pattern_ids": list(pattern_ids or []),
        "givens81_values": list(givens81_values or []),
        "solution81_values": list(solution81_values or []),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk cleanup wrapper for raw generation pool candidates.jsonl."
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
        "--all-lines",
        action="store_true",
        help="Target all raw lines in candidates.jsonl.",
    )
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
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--confirm", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    jsonl_path = Path(args.jsonl)

    selector = _select_generation_pool_targets(
        jsonl_path=jsonl_path,
        all_lines=bool(args.all_lines),
        line_numbers=args.line_numbers,
        generation_seeds=args.generation_seeds,
        pattern_ids=args.pattern_ids,
        givens81_values=args.givens81_values,
        solution81_values=args.solution81_values,
    )

    has_selector = any(selector.values())
    if not has_selector:
        print("No generation-pool selector was supplied.", flush=True)
        return 1

    is_delete_all = bool(args.all_lines)
    selected_count_hint = len(selector["line_numbers"]) if selector["line_numbers"] else 0

    if args.dry_run:
        return run_delete_candidates_jsonl(
            jsonl_path=jsonl_path,
            records_dir=Path(args.records_dir),
            backup_root=Path(args.backup_root),
            report_dir=Path(args.report_dir),
            archive_removed_to=Path(args.archive_removed_to) if args.archive_removed_to else None,
            line_numbers=selector["line_numbers"],
            generation_seeds=selector["generation_seeds"],
            pattern_ids=selector["pattern_ids"],
            givens81_values=selector["givens81_values"],
            solution81_values=selector["solution81_values"],
            dry_run=True,
            yes=bool(args.yes),
            confirm=args.confirm,
        )

    first_plan = analyze_candidate_jsonl_delete(
        jsonl_path=jsonl_path,
        requested_action=DeleteAction.RAW_DELETE,
        line_numbers=selector["line_numbers"],
        generation_seeds=selector["generation_seeds"],
        pattern_ids=selector["pattern_ids"],
        givens81_values=selector["givens81_values"],
        solution81_values=selector["solution81_values"],
        records_dir=Path(args.records_dir),
    )

    is_batch = False
    confirmed = is_confirmation_satisfied(
        plan=first_plan,
        provided_token=args.confirm,
        yes=bool(args.yes),
        is_batch=is_batch,
        is_delete_all=is_delete_all,
    )
    if not confirmed:
        spec = confirmation_token_for_plan(
            first_plan,
            is_batch=is_batch,
            is_delete_all=is_delete_all,
        )
        print(f"Selected raw-candidate line count hint: {selected_count_hint or 'computed in preview'}", flush=True)
        print(f"Expected confirmation token: {spec.required_token}", flush=True)
        confirmed = prompt_for_confirmation(
            plan=first_plan,
            is_batch=is_batch,
            is_delete_all=is_delete_all,
        )

    if not confirmed:
        print("Confirmation failed. No mutation was performed.", flush=True)
        return 1

    return run_delete_candidates_jsonl(
        jsonl_path=jsonl_path,
        records_dir=Path(args.records_dir),
        backup_root=Path(args.backup_root),
        report_dir=Path(args.report_dir),
        archive_removed_to=Path(args.archive_removed_to) if args.archive_removed_to else None,
        line_numbers=selector["line_numbers"],
        generation_seeds=selector["generation_seeds"],
        pattern_ids=selector["pattern_ids"],
        givens81_values=selector["givens81_values"],
        solution81_values=selector["solution81_values"],
        dry_run=False,
        yes=True,
        confirm=args.confirm,
    )


if __name__ == "__main__":
    raise SystemExit(main())