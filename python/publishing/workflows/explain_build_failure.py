from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.book_builder.book_spec_loader import load_book_spec
from python.publishing.book_builder.capacity_analyzer import analyze_book_capacity, explain_capacity_failure
from python.publishing.inventory.library_inventory_store import load_library_inventory
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain why a book spec cannot currently be built.")
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
    print("explain_build_failure.py", flush=True)
    print("=" * 72, flush=True)

    for line in explain_capacity_failure(report):
        print(line, flush=True)

    print("=" * 72, flush=True)
    return 0 if report["buildable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())