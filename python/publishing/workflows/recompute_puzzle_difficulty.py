from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.puzzle_catalog.catalog_index import load_catalog_index, save_catalog_index
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir
from python.publishing.puzzle_catalog.difficulty_enricher import enrich_candidate_difficulty
from python.publishing.puzzle_catalog.metadata_enricher import build_print_header


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute techniques_difficulty and puzzle_difficulty for puzzle records in a directory."
    )
    parser.add_argument(
        "--input-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Directory containing puzzle record JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)

    records = load_puzzle_records_from_dir(input_dir)
    index = load_catalog_index(input_dir)

    updated_count = 0

    for record in records:
        payload = enrich_candidate_difficulty(
            techniques_used=record.techniques_used,
        )
        record.techniques_difficulty = list(payload["techniques_difficulty"])
        record.puzzle_difficulty = str(payload["puzzle_difficulty"])
        record.difficulty_version = str(payload["difficulty_version"])

        display_code = ""
        if getattr(record, "print_header", None) is not None:
            display_code = str(record.print_header.display_code or "").strip()
        if not display_code:
            display_code = str(record.local_puzzle_code or record.record_id)

        record.print_header = build_print_header(
            display_code=display_code,
            difficulty_label=record.puzzle_difficulty,
            weight=int(record.weight),
        )

        path = input_dir / f"{record.record_id}.json"
        path.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        entry = index.get("records_by_id", {}).get(record.record_id)
        if entry is not None:
            entry["puzzle_difficulty"] = record.puzzle_difficulty
            entry["difficulty_version"] = record.difficulty_version
            entry["updated_at"] = record.updated_at

        updated_count += 1

    index_path = save_catalog_index(index, input_dir)

    print("=" * 72, flush=True)
    print("recompute_puzzle_difficulty.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Updated records: {updated_count}", flush=True)
    print(f"Directory:       {input_dir}", flush=True)
    print(f"Catalog index:   {index_path}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())