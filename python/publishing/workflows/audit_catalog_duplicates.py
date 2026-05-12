from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir
from python.publishing.puzzle_catalog.solution_signature import build_solution_signature


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a puzzle-record catalog directory for exact/substitution-equivalent duplicate solutions."
    )
    parser.add_argument(
        "--input-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Directory containing puzzle record JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default="datasets/sudoku_books/classic9/puzzle_records/_duplicate_audit.json",
        help="Where to write the audit report JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)

    records = load_puzzle_records_from_dir(input_dir)

    groups = defaultdict(list)
    for record in records:
        signature = build_solution_signature(record.solution81)
        groups[signature].append(
            {
                "record_id": record.record_id,
                "title": record.title,
                "weight": record.weight,
                "pattern_id": record.pattern_id,
                "candidate_status": record.candidate_status,
            }
        )

    duplicate_groups = []
    for signature, members in groups.items():
        if len(members) > 1:
            duplicate_groups.append(
                {
                    "solution_signature": signature,
                    "count": len(members),
                    "members": members,
                }
            )

    report = {
        "schema_version": 1,
        "catalog_record_count": len(records),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72, flush=True)
    print("audit_catalog_duplicates.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Catalog record count:   {len(records)}", flush=True)
    print(f"Duplicate group count: {len(duplicate_groups)}", flush=True)
    print(f"Report written:        {output_json}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())