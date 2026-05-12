from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.inventory.library_inventory_store import load_library_inventory
from python.publishing.inventory.removal_guard import can_remove_record_from_catalog


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a catalog record can be safely removed.")
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
        help="Directory containing _library_inventory.json",
    )
    parser.add_argument(
        "--library-id",
        required=True,
        help="Library id.",
    )
    parser.add_argument(
        "--record-id",
        required=True,
        help="Catalog record id to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = load_library_inventory(
        base_dir=Path(args.inventory_dir),
        library_id=args.library_id,
    )

    ok, reasons = can_remove_record_from_catalog(
        inventory,
        record_id=args.record_id,
    )

    print("=" * 72, flush=True)
    print("check_record_removal.py", flush=True)
    print("=" * 72, flush=True)
    print(f"record_id: {args.record_id}", flush=True)
    print(f"removable: {'yes' if ok else 'no'}", flush=True)
    if reasons:
        print("-" * 72, flush=True)
        for reason in reasons:
            print(f"* {reason}", flush=True)
    print("=" * 72, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())