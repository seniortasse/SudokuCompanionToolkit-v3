from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.puzzle_catalog.pattern_production_job import (
    load_pattern_production_job,
    run_pattern_production_job,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSON-defined pattern-driven puzzle production job."
    )
    parser.add_argument(
        "--job",
        required=True,
        help="Path to a JSON puzzle production job spec.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    job_path = Path(args.job)
    job_spec = load_pattern_production_job(job_path)
    result = run_pattern_production_job(job_spec)

    print(f"Job id:             {result['job_id']}", flush=True)
    print(f"Selected patterns:  {len(result['selected_patterns'])}", flush=True)
    print(f"Production requests:{len(result['requests'])}", flush=True)
    print(f"Candidates written: {len(result['candidates'])}", flush=True)
    print(f"Rejected requests:  {len(result['rejected'])}", flush=True)
    print(f"Candidates JSONL:   {result['output_jsonl']}", flush=True)
    print(f"Input workbook:     {result['request_workbook']}", flush=True)
    print(f"Output workbook:    {result['output_workbook']}", flush=True)
    print(f"Run report:         {result['report_path']}", flush=True)
    print(f"STDOUT log:         {result['stdout_path']}", flush=True)
    print(f"STDERR log:         {result['stderr_path']}", flush=True)
    print(f"Rejected log:       {result['rejected_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())