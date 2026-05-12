from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.inventory.library_inventory_store import load_library_inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List library inventory assignments.")
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
        help="Directory containing _library_inventory.json",
    )
    parser.add_argument(
        "--library-id",
        required=True,
        help="Library id to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = load_library_inventory(
        base_dir=Path(args.inventory_dir),
        library_id=args.library_id,
    )

    records = dict(inventory.get("records", {}))
    assigned = [entry for entry in records.values() if int(entry.get("assignment_count", 0)) > 0]

    print("=" * 72, flush=True)
    print("list_library_inventory.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Library id:       {inventory.get('library_id')}", flush=True)
    print(f"Tracked records:  {len(records)}", flush=True)
    print(f"Assigned records: {len(assigned)}", flush=True)
    print("-" * 72, flush=True)

    for entry in sorted(assigned, key=lambda x: str(x.get("record_id"))):
        print(
            f"{entry.get('record_id')} | status={entry.get('candidate_status')} | "
            f"assignments={entry.get('assignment_count')} | last_book={entry.get('last_book_id')}",
            flush=True,
        )

    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())