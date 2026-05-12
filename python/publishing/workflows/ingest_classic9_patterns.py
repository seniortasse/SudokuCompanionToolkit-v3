from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from python.publishing.ids.id_policy import build_library_id
from python.publishing.pattern_library.ingest_excel_patterns import (
    build_sources_from_workbook,
    ingest_sources_into_registry,
)
from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)


DEFAULT_LIBRARY_SHORT = "CL9"
DEFAULT_LIBRARY_ID = build_library_id(DEFAULT_LIBRARY_SHORT)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest classic 9x9 pattern Excel files into the canonical pattern catalog/store."
    )
    parser.add_argument(
        "--workbook",
        action="append",
        required=True,
        help="Path to an Excel workbook. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        default=None,
        help="Optional sheet name filter. Can be supplied multiple times. If omitted, all sheets are used.",
    )
    parser.add_argument(
        "--top-row",
        type=int,
        default=1,
        help="Top row of the 9x9 pattern block. Default: 1",
    )
    parser.add_argument(
        "--left-col",
        type=int,
        default=1,
        help="Left column of the 9x9 pattern block. Default: 1",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional tag to apply to all ingested patterns from this run.",
    )
    parser.add_argument(
        "--patterns-dir",
        default="datasets/sudoku_books/classic9/patterns",
        help="Output pattern directory.",
    )
    parser.add_argument(
        "--library-id",
        default=DEFAULT_LIBRARY_ID,
        help="Canonical library id. Default: LIB-CL9",
    )
    parser.add_argument(
        "--library-short",
        default=DEFAULT_LIBRARY_SHORT,
        help="Library short code used in ids. Default: CL9",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    patterns_dir = Path(args.patterns_dir)
    patterns_dir.mkdir(parents=True, exist_ok=True)

    registry = load_pattern_store(patterns_dir)
    if not registry.library_id:
        registry.library_id = args.library_id

    all_sources = []
    for workbook_arg in args.workbook:
        workbook_path = Path(workbook_arg)
        if not workbook_path.exists():
            print(f"ERROR: workbook not found: {workbook_path}", flush=True)
            return 1

        sources = build_sources_from_workbook(
            workbook_path=workbook_path,
            sheet_names=args.sheet,
            top_row=args.top_row,
            left_col=args.left_col,
            default_tags=args.tag,
        )
        all_sources.extend(sources)

    timestamp = _now_iso()
    result = ingest_sources_into_registry(
        library_id=registry.library_id or args.library_id,
        library_short=args.library_short,
        sources=all_sources,
        registry=registry,
        created_at=timestamp,
        updated_at=timestamp,
    )

    paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)

    for pattern in result.added_patterns:
        pattern_path = patterns_dir / f"{pattern.pattern_id}.json"
        pattern_path.write_text(
            json.dumps(pattern.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Workbooks processed:  {len(args.workbook)}", flush=True)
    print(f"Sheets ingested:      {len(all_sources)}", flush=True)
    print(f"Added patterns:       {len(result.added_patterns)}", flush=True)
    print(f"Skipped duplicates:   {len(result.skipped_duplicates)}", flush=True)
    print(f"Validation failures:  {len(result.validation_failures)}", flush=True)
    print(f"Catalog path:         {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Registry path:        {paths['registry']}", flush=True)
    print(f"Index by id:          {paths['by_id']}", flush=True)
    print(f"Index by mask:        {paths['by_mask']}", flush=True)
    print(f"Index by family:      {paths['by_family']}", flush=True)

    if result.skipped_duplicates:
        print("\nSkipped duplicate patterns:", flush=True)
        for name in result.skipped_duplicates:
            print(f"  - {name}", flush=True)

    if result.validation_failures:
        print("\nValidation failures:", flush=True)
        for pattern_name, errors in result.validation_failures:
            print(f"  - {pattern_name}", flush=True)
            for error in errors:
                print(f"      * {error}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())