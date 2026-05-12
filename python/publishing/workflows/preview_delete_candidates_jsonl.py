from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.cleanup import (
    DeleteAction,
    analyze_candidate_jsonl_delete,
    decide_candidate_jsonl_delete,
    render_delete_plan,
    render_policy_decision,
    render_report_summary,
    write_delete_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview raw candidate deletion from candidates.jsonl."
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
        "--report-dir",
        default="runs/publishing/delete_reports",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    plan = analyze_candidate_jsonl_delete(
        jsonl_path=Path(args.jsonl),
        requested_action=DeleteAction.RAW_DELETE,
        line_numbers=args.line_numbers,
        generation_seeds=args.generation_seeds,
        pattern_ids=args.pattern_ids,
        givens81_values=args.givens81_values,
        solution81_values=args.solution81_values,
        records_dir=Path(args.records_dir),
    )
    decision = decide_candidate_jsonl_delete(plan)

    print(render_delete_plan(plan), flush=True)
    print("", flush=True)
    print(render_policy_decision(decision), flush=True)

    report_path = write_delete_report(
        plan=plan,
        decision=decision,
        report_dir=Path(args.report_dir),
        stage="preview",
        extra={
            "workflow": "preview_delete_candidates_jsonl",
            "jsonl_path": str(args.jsonl),
            "selector": {
                "line_numbers": args.line_numbers or [],
                "generation_seeds": args.generation_seeds or [],
                "pattern_ids": args.pattern_ids or [],
                "givens81_values": args.givens81_values or [],
                "solution81_values": args.solution81_values or [],
            },
        },
    )
    print("", flush=True)
    print(render_report_summary(report_path, stage="preview"), flush=True)

    return 1 if decision.outcome == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())