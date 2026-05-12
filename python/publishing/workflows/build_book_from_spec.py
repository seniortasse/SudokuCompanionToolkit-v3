from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from python.publishing.book_builder.book_manifest_builder import build_book_from_spec
from python.publishing.book_builder.book_spec_loader import load_book_spec
from python.publishing.book_builder.capacity_analyzer import analyze_book_capacity, explain_capacity_failure
from python.publishing.inventory.library_inventory_store import load_library_inventory
from python.publishing.puzzle_catalog.catalog_index import load_catalog_index, save_catalog_index, update_record_status
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir
from python.publishing.qc.validate_book_manifest import validate_book_manifest


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a book manifest and section manifests from a declarative book spec."
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to the book spec JSON file.",
    )
    parser.add_argument(
        "--puzzle-records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Directory containing canonical puzzle record JSON files.",
    )
    parser.add_argument(
        "--patterns-dir",
        default="datasets/sudoku_books/classic9/patterns",
        help="Directory containing canonical pattern catalog/store files.",
    )
    parser.add_argument(
        "--output-books-dir",
        default="datasets/sudoku_books/classic9/books",
        help="Base output directory for built books.",
    )
    parser.add_argument(
        "--inventory-dir",
        default="datasets/sudoku_books/classic9/catalogs",
        help="Directory where the library inventory ledger should be stored.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    spec_path = Path(args.spec)
    puzzle_records_dir = Path(args.puzzle_records_dir)
    patterns_dir = Path(args.patterns_dir)
    output_books_dir = Path(args.output_books_dir)
    inventory_dir = Path(args.inventory_dir)

    _log("=" * 72)
    _log("build_book_from_spec.py starting")
    _log("=" * 72)
    _log(f"Spec path:            {spec_path.resolve()}")
    _log(f"Puzzle records dir:   {puzzle_records_dir.resolve()}")
    _log(f"Patterns dir:         {patterns_dir.resolve()}")
    _log(f"Output books dir:     {output_books_dir.resolve()}")
    _log(f"Inventory dir:        {inventory_dir.resolve()}")
    _log("=" * 72)

    if not spec_path.exists():
        _log(f"ERROR: book spec not found: {spec_path}")
        return 1

    if not puzzle_records_dir.exists():
        _log(f"ERROR: puzzle records directory not found: {puzzle_records_dir}")
        return 1

    spec = load_book_spec(spec_path)
    puzzle_records = load_puzzle_records_from_dir(puzzle_records_dir)

    _log(f"Loaded puzzle records: {len(puzzle_records)}")
    _log(f"Book id:               {spec.book_id}")
    _log(f"Sections in spec:      {len(spec.sections)}")

    timestamp = _now_iso()

    try:
        built = build_book_from_spec(
            spec=spec,
            puzzle_records=puzzle_records,
            inventory_dir=inventory_dir,
            patterns_dir=patterns_dir,
            created_at=timestamp,
            updated_at=timestamp,
        )
    except Exception as exc:
        _log(f"ERROR while building book: {exc}")

        try:
            library_inventory = load_library_inventory(
                base_dir=inventory_dir,
                library_id=spec.library_id,
            )
            capacity = analyze_book_capacity(
                spec=spec,
                puzzle_records=puzzle_records,
                library_inventory=library_inventory,
            )
            explanation = explain_capacity_failure(
                spec=spec,
                capacity=capacity,
            )
            _log("-" * 72)
            _log("Capacity analysis:")
            _log(explanation)
            _log("-" * 72)
        except Exception as capacity_exc:
            _log(f"WARNING: capacity analysis also failed: {capacity_exc}")

        return 1

    errors = validate_book_manifest(built.book_manifest)
    if errors:
        _log("ERROR: built book manifest failed validation")
        for error in errors:
            _log(f"  * {error}")
        return 1

    book_dir = output_books_dir / spec.book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    book_manifest_path = book_dir / "book_manifest.json"
    sections_dir = book_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    puzzles_dir = book_dir / "puzzles"
    puzzles_dir.mkdir(parents=True, exist_ok=True)

    _write_json(book_manifest_path, built.book_manifest.to_dict())

    section_manifest_paths = []
    for section_manifest in built.section_manifests:
        path = sections_dir / f"{section_manifest.section_code}.json"
        _write_json(path, section_manifest.to_dict())
        section_manifest_paths.append(path)

    puzzle_paths = []
    for record in built.assigned_puzzles:
        path = puzzles_dir / f"{record.record_id}.json"
        _write_json(path, record.to_dict())
        puzzle_paths.append(path)

    try:
        catalog_index = load_catalog_index(puzzle_records_dir)
        for record in built.assigned_puzzles:
            update_record_status(
                catalog_index,
                record_id=record.record_id,
                candidate_status="assigned",
            )
        catalog_index_path = save_catalog_index(catalog_index, puzzle_records_dir)
    except Exception as exc:
        _log(f"ERROR while updating puzzle catalog index: {exc}")
        return 1

    summary = {
        "timestamp": timestamp,
        "book_id": built.book_manifest.book_id,
        "section_count": len(built.section_manifests),
        "puzzle_count": len(built.assigned_puzzles),
        "inventory_path": built.inventory_path,
        "book_manifest_path": str(book_manifest_path),
        "section_manifest_paths": [str(path) for path in section_manifest_paths],
        "puzzle_paths": [str(path) for path in puzzle_paths],
        "catalog_index_path": str(catalog_index_path),
        "patterns_dir": str(patterns_dir),
    }
    summary_path = book_dir / "_build_summary.json"
    _write_json(summary_path, summary)

    _log(f"Book manifest written:  {book_manifest_path}")
    _log(f"Section manifests:      {len(section_manifest_paths)}")
    _log(f"Assigned puzzles:       {len(puzzle_paths)}")
    _log(f"Catalog index updated:  {catalog_index_path}")
    if built.inventory_path:
        _log(f"Inventory updated:      {built.inventory_path}")
    _log(f"Build summary written:  {summary_path}")
    _log("=" * 72)
    _log("build_book_from_spec.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())