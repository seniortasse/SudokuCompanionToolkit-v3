from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.book_builder.book_package_store import load_built_book_package, save_built_book_package
from python.publishing.book_builder.ordering import order_section_puzzles
from python.publishing.book_builder.puzzle_selector import select_puzzles_for_section
from python.publishing.inventory.assignment_ledger import register_assignment, unregister_assignment_for_record
from python.publishing.inventory.assignment_rules import filter_records_available_for_library
from python.publishing.inventory.library_inventory_store import load_library_inventory, save_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index, save_catalog_index, update_record_status
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace one puzzle inside a built book section.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--section-code", required=True)
    parser.add_argument("--record-id", required=True, help="Current record_id to replace")
    parser.add_argument(
        "--books-dir",
        default="datasets/sudoku_books/classic9/books",
    )
    parser.add_argument(
        "--records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
    )
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
    )
    parser.add_argument("--library-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    book_dir = Path(args.books_dir) / args.book_id
    book_manifest, section_manifests, assigned_puzzles = load_built_book_package(book_dir)

    target_section = None
    for section in section_manifests:
        if section.section_code == args.section_code:
            target_section = section
            break
    if target_section is None:
        print(f"Section not found: {args.section_code}", flush=True)
        return 1

    current_record = None
    for record in assigned_puzzles:
        if record.record_id == args.record_id and record.section_code == args.section_code:
            current_record = record
            break
    if current_record is None:
        print(f"Record {args.record_id} not found in book {args.book_id} section {args.section_code}", flush=True)
        return 1

    inventory = load_library_inventory(
        base_dir=Path(args.inventory_dir),
        library_id=args.library_id,
    )

    all_records = load_puzzle_records_from_dir(Path(args.records_dir))
    available_records = filter_records_available_for_library(
        records=all_records,
        inventory=inventory,
    )

    eligible = select_puzzles_for_section(
        puzzle_records=available_records,
        section_criteria=target_section.criteria.to_dict(),
        global_filters=book_manifest.global_filters,
    )

    eligible = [r for r in eligible if r.record_id != current_record.record_id]
    eligible = order_section_puzzles(
        eligible,
        ordering_policy=book_manifest.ordering_policy,
    )

    if not eligible:
        print("No eligible replacement puzzle found for this section.", flush=True)
        return 1

    replacement_source = eligible[0]

    replacement = replacement_source
    replacement.book_id = current_record.book_id
    replacement.aisle_id = current_record.aisle_id
    replacement.section_id = current_record.section_id
    replacement.section_code = current_record.section_code
    replacement.local_puzzle_code = current_record.local_puzzle_code
    replacement.friendly_puzzle_id = current_record.friendly_puzzle_id
    replacement.puzzle_uid = current_record.puzzle_uid
    replacement.position_in_section = current_record.position_in_section
    replacement.position_in_book = current_record.position_in_book
    replacement.print_header.display_code = current_record.print_header.display_code
    replacement.candidate_status = "assigned"

    updated_puzzles = []
    for record in assigned_puzzles:
        if record.record_id == current_record.record_id and record.section_code == current_record.section_code:
            updated_puzzles.append(replacement)
        else:
            updated_puzzles.append(record)

    unregister_assignment_for_record(
        inventory,
        record_id=current_record.record_id,
        book_id=args.book_id,
        section_id=current_record.section_id,
        puzzle_uid=current_record.puzzle_uid,
    )
    register_assignment(
        inventory,
        record=replacement,
    )
    save_library_inventory(
        inventory=inventory,
        base_dir=Path(args.inventory_dir),
    )

    catalog_index = load_catalog_index(Path(args.records_dir))
    update_record_status(catalog_index, record_id=current_record.record_id, candidate_status="available")
    update_record_status(catalog_index, record_id=replacement.record_id, candidate_status="assigned")
    save_catalog_index(catalog_index, Path(args.records_dir))

    target_section.puzzle_ids = [
        replacement.puzzle_uid if pid == current_record.puzzle_uid else pid
        for pid in target_section.puzzle_ids
    ]

    save_built_book_package(
        book_dir=book_dir,
        book_manifest=book_manifest,
        section_manifests=section_manifests,
        assigned_puzzles=updated_puzzles,
    )

    print("=" * 72, flush=True)
    print("replace_book_puzzle.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Book id:        {args.book_id}", flush=True)
    print(f"Section code:   {args.section_code}", flush=True)
    print(f"Old record_id:  {current_record.record_id}", flush=True)
    print(f"New record_id:  {replacement.record_id}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())