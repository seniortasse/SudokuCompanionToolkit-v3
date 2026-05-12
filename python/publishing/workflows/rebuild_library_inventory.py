from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.inventory.assignment_ledger import register_assignment
from python.publishing.inventory.library_inventory_store import save_library_inventory
from python.publishing.schemas.models import PuzzleRecord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the library inventory ledger by scanning built book puzzle JSON files."
    )
    parser.add_argument(
        "--books-dir",
        default="datasets/sudoku_books/classic9/books",
        help="Directory containing built book folders.",
    )
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
        help="Directory where _library_inventory.json should be written.",
    )
    parser.add_argument(
        "--library-id",
        required=True,
        help="Library id.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    books_dir = Path(args.books_dir)
    inventory_dir = Path(args.inventory_dir)

    inventory = {
        "schema_version": 1,
        "library_id": args.library_id,
        "records": {},
    }

    scanned = 0
    assigned = 0

    if books_dir.exists():
        for puzzle_path in books_dir.glob("*/puzzles/*.json"):
            data = json.loads(puzzle_path.read_text(encoding="utf-8"))
            record = PuzzleRecord.from_dict(data)
            scanned += 1

            if record.library_id != args.library_id:
                continue
            if not record.book_id:
                continue

            register_assignment(
                inventory,
                record=record,
            )
            assigned += 1

    path = save_library_inventory(
        inventory=inventory,
        base_dir=inventory_dir,
    )

    print("=" * 72, flush=True)
    print("rebuild_library_inventory.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Scanned puzzle files: {scanned}", flush=True)
    print(f"Registered entries:   {assigned}", flush=True)
    print(f"Inventory path:       {path}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())