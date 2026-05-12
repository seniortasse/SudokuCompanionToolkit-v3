from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from python.publishing.app_catalog_builder.catalog_manifest_builder import (
    DEFAULT_CLASSIC9_AISLES,
    build_aisle_manifests,
    build_catalog_manifest,
    build_library_manifest,
)
from python.publishing.app_catalog_builder.compact_export import export_compact_app_catalog
from python.publishing.app_catalog_builder.search_index_builder import (
    build_book_by_title_index,
    build_puzzles_by_pattern_index,
    build_puzzles_by_technique_index,
    build_puzzles_by_weight_band_index,
)
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir


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
        description="Build the classic9 catalog layer from canonical puzzle records."
    )
    parser.add_argument(
        "--puzzle-records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Directory containing canonical puzzle record JSON files.",
    )
    parser.add_argument(
        "--catalogs-dir",
        default="datasets/sudoku_books/classic9/catalogs",
        help="Directory where catalog manifests will be written.",
    )
    parser.add_argument(
        "--search-indexes-dir",
        default="datasets/sudoku_books/classic9/search_indexes",
        help="Directory where search indexes will be written.",
    )
    parser.add_argument(
        "--app-export-dir",
        default="exports/sudoku_books/app_catalog/classic9",
        help="Directory for compact app catalog export.",
    )
    parser.add_argument(
        "--catalog-version",
        default="1.0.0",
        help="Catalog version string.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    puzzle_records_dir = Path(args.puzzle_records_dir)
    catalogs_dir = Path(args.catalogs_dir)
    search_indexes_dir = Path(args.search_indexes_dir)
    app_export_dir = Path(args.app_export_dir)

    _log("=" * 72)
    _log("build_classic9_catalog.py starting")
    _log("=" * 72)
    _log(f"Puzzle records dir:   {puzzle_records_dir.resolve()}")
    _log(f"Catalogs dir:         {catalogs_dir.resolve()}")
    _log(f"Search indexes dir:   {search_indexes_dir.resolve()}")
    _log(f"App export dir:       {app_export_dir.resolve()}")
    _log(f"Catalog version:      {args.catalog_version}")
    _log("=" * 72)

    if not puzzle_records_dir.exists():
        _log(f"ERROR: puzzle records directory does not exist: {puzzle_records_dir}")
        return 1

    puzzle_records = load_puzzle_records_from_dir(puzzle_records_dir)
    if not puzzle_records:
        _log("ERROR: no puzzle records were found. Build puzzle records first.")
        return 1

    _log(f"Loaded puzzle records: {len(puzzle_records)}")

    timestamp = _now_iso()

    library_manifest = build_library_manifest(
        library_id="LIB-CL9",
        slug="classic9",
        title="Classic Sudoku Library",
        subtitle="Standard 9x9 Sudoku catalog",
        description="The classic 9x9 Sudoku library used by the app and by printable book production.",
        layout_type="classic9x9",
        grid_size=9,
        charset="123456789",
        box_schema="3x3",
        status="active",
        aisle_definitions=DEFAULT_CLASSIC9_AISLES,
        created_at=timestamp,
        updated_at=timestamp,
    )

    aisle_manifests = build_aisle_manifests(
        library_id=library_manifest.library_id,
        puzzle_records=puzzle_records,
        aisle_definitions=DEFAULT_CLASSIC9_AISLES,
        created_at=timestamp,
        updated_at=timestamp,
    )

    _log(f"Built aisle manifests: {len(aisle_manifests)}")

    puzzles_by_technique = build_puzzles_by_technique_index(puzzle_records)
    puzzles_by_weight_band = build_puzzles_by_weight_band_index(puzzle_records)
    puzzles_by_pattern = build_puzzles_by_pattern_index(puzzle_records)
    books_by_title = build_book_by_title_index(puzzle_records)

    search_indexes: Dict[str, Dict[str, object]] = {
        "puzzles_by_technique": puzzles_by_technique,
        "puzzles_by_weight_band": puzzles_by_weight_band,
        "puzzles_by_pattern": puzzles_by_pattern,
        "books_by_title": books_by_title,
    }

    catalog_manifest = build_catalog_manifest(
        catalog_version=args.catalog_version,
        generated_at=timestamp,
        library_manifest=library_manifest,
        puzzle_records=puzzle_records,
        index_files={
            "books_by_title": "indexes/books_by_title.json",
            "puzzles_by_technique": "indexes/puzzles_by_technique.json",
            "puzzles_by_weight_band": "indexes/puzzles_by_weight_band.json",
            "puzzles_by_pattern": "indexes/puzzles_by_pattern.json",
        },
    )

    catalogs_dir.mkdir(parents=True, exist_ok=True)
    search_indexes_dir.mkdir(parents=True, exist_ok=True)

    _write_json(catalogs_dir / "catalog_manifest.json", catalog_manifest.to_dict())

    for aisle_manifest in aisle_manifests:
        _write_json(catalogs_dir / f"{aisle_manifest.aisle_id}.json", aisle_manifest.to_dict())

    for index_name, payload in search_indexes.items():
        _write_json(search_indexes_dir / f"{index_name}.json", payload)

    export_compact_app_catalog(
        export_dir=app_export_dir,
        catalog_manifest=catalog_manifest,
        aisle_manifests=aisle_manifests,
        puzzle_records=puzzle_records,
        search_indexes=search_indexes,
    )

    _log("-" * 72)
    _log("Catalog files written:")
    _log(f"  + {catalogs_dir / 'catalog_manifest.json'}")
    for aisle_manifest in aisle_manifests:
        _log(f"  + {catalogs_dir / f'{aisle_manifest.aisle_id}.json'}")

    _log("Search index files written:")
    for index_name in search_indexes.keys():
        _log(f"  + {search_indexes_dir / f'{index_name}.json'}")

    _log(f"Compact app export written to: {app_export_dir}")
    _log("=" * 72)
    _log("build_classic9_catalog.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())