from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.book_builder.book_spec_loader import load_book_spec
from python.publishing.book_builder.capacity_analyzer import analyze_book_capacity, explain_capacity_failure
from python.publishing.inventory.library_inventory_store import load_library_inventory
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report whether a book spec has enough eligible puzzles to build.")
    parser.add_argument("--spec", required=True, help="Path to the book spec JSON file.")
    parser.add_argument(
        "--puzzle-records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Directory containing canonical puzzle record JSON files.",
    )
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
        help="Directory containing the library inventory ledger.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path where the capacity report JSON should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    spec = load_book_spec(Path(args.spec))
    puzzle_records = load_puzzle_records_from_dir(Path(args.puzzle_records_dir))
    library_inventory = load_library_inventory(
        base_dir=Path(args.inventory_dir),
        library_id=spec.library_id,
    )

    report = analyze_book_capacity(
        spec=spec,
        puzzle_records=puzzle_records,
        library_inventory=library_inventory,
    )

    print("=" * 72, flush=True)
    print("capacity_report.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Book id:      {report['book_id']}", flush=True)
    print(f"Library id:   {report['library_id']}", flush=True)
    print(f"Buildable:    {'yes' if report['buildable'] else 'no'}", flush=True)
    print("-" * 72, flush=True)

    for section in report["sections"]:
        print(
            f"{section['section_code']} | requested={section['requested']} | "
            f"eligible={section['eligible']} | shortage={section['shortage']}",
            flush=True,
        )

    if not report["buildable"]:
        print("-" * 72, flush=True)
        for line in explain_capacity_failure(report):
            print(f"* {line}", flush=True)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("-" * 72, flush=True)
        print(f"Report JSON: {output_path}", flush=True)

    print("=" * 72, flush=True)
    return 0 if report["buildable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())