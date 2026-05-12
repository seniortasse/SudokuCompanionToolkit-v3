from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.book_builder.book_package_store import load_built_book_package, save_built_book_package
from python.publishing.book_builder.ordering import order_section_puzzles
from python.publishing.book_builder.puzzle_selector import select_puzzles_for_section
from python.publishing.inventory.assignment_ledger import (
    register_assignment,
    unregister_assignment_for_record,
)
from python.publishing.inventory.assignment_rules import filter_records_available_for_library
from python.publishing.inventory.library_inventory_store import load_library_inventory, save_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index, save_catalog_index, update_record_status
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild one section of a built book from its criteria.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--section-code", required=True)
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--books-dir", default="datasets/sudoku_books/classic9/books")
    parser.add_argument("--records-dir", default="datasets/sudoku_books/classic9/puzzle_records")
    parser.add_argument("--inventory-dir", default="datasets/sudoku_books/classic9/catalogs")
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

    section_records = [r for r in assigned_puzzles if r.section_code == args.section_code]
    other_records = [r for r in assigned_puzzles if r.section_code != args.section_code]

    inventory = load_library_inventory(
        base_dir=Path(args.inventory_dir),
        library_id=args.library_id,
    )

    for record in section_records:
        unregister_assignment_for_record(
            inventory,
            record_id=record.record_id,
            book_id=args.book_id,
            section_id=record.section_id,
            puzzle_uid=record.puzzle_uid,
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
    eligible = order_section_puzzles(
        eligible,
        ordering_policy=book_manifest.ordering_policy,
    )

    if len(eligible) < target_section.puzzle_count:
        print(
            f"Not enough eligible records to rebuild section {args.section_code}: "
            f"need {target_section.puzzle_count}, found {len(eligible)}",
            flush=True,
        )
        return 1

    rebuilt_section_records = []
    chosen_sources = eligible[: target_section.puzzle_count]

    for idx, source in enumerate(chosen_sources, start=1):
        old = section_records[idx - 1] if idx - 1 < len(section_records) else section_records[-1]
        source.book_id = old.book_id
        source.aisle_id = old.aisle_id
        source.section_id = old.section_id
        source.section_code = old.section_code
        source.local_puzzle_code = old.local_puzzle_code
        source.friendly_puzzle_id = old.friendly_puzzle_id
        source.puzzle_uid = old.puzzle_uid
        source.position_in_section = old.position_in_section
        source.position_in_book = old.position_in_book
        source.print_header.display_code = old.print_header.display_code
        source.candidate_status = "assigned"
        rebuilt_section_records.append(source)

    for record in rebuilt_section_records:
        register_assignment(
            inventory,
            record=record,
        )
    save_library_inventory(
        inventory=inventory,
        base_dir=Path(args.inventory_dir),
    )

    catalog_index = load_catalog_index(Path(args.records_dir))
    for record in section_records:
        update_record_status(catalog_index, record_id=record.record_id, candidate_status="available")
    for record in rebuilt_section_records:
        update_record_status(catalog_index, record_id=record.record_id, candidate_status="assigned")
    save_catalog_index(catalog_index, Path(args.records_dir))

    target_section.puzzle_ids = [record.puzzle_uid for record in rebuilt_section_records if record.puzzle_uid]

    updated_puzzles = other_records + rebuilt_section_records
    updated_puzzles = sorted(
        updated_puzzles,
        key=lambda r: (r.position_in_book if r.position_in_book is not None else 10**9, r.record_id),
    )

    save_built_book_package(
        book_dir=book_dir,
        book_manifest=book_manifest,
        section_manifests=section_manifests,
        assigned_puzzles=updated_puzzles,
    )

    print("=" * 72, flush=True)
    print("rebuild_section.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Book id:      {args.book_id}", flush=True)
    print(f"Section code: {args.section_code}", flush=True)
    print(f"Rebuilt with: {len(rebuilt_section_records)} puzzle(s)", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())