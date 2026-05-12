from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.pattern_library.pattern_generator import (
    available_generator_families,
    generate_patterns_into_registry,
)
from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate artistic pattern records directly into the canonical pattern catalog."
    )
    parser.add_argument(
        "--patterns-dir",
        default="datasets/sudoku_books/classic9/patterns",
        help="Pattern catalog directory.",
    )
    parser.add_argument(
        "--family",
        action="append",
        required=True,
        help=f"Pattern family to generate. Available: {', '.join(available_generator_families())}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of patterns to attempt to generate. Default: 1",
    )
    parser.add_argument(
        "--library-short",
        default="CL9",
        help="Library short code used in generated pattern ids. Default: CL9",
    )
    parser.add_argument(
        "--author",
        default="system",
        help="Author label for generated patterns. Default: system",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    patterns_dir = Path(args.patterns_dir)
    patterns_dir.mkdir(parents=True, exist_ok=True)

    registry = load_pattern_store(patterns_dir)

    result = generate_patterns_into_registry(
        registry=registry,
        library_short=args.library_short,
        family_ids=args.family,
        count=args.count,
        author=args.author,
    )

    paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)

    for pattern in result.added_patterns:
        pattern_path = patterns_dir / f"{pattern.pattern_id}.json"
        pattern_path.write_text(
            json.dumps(pattern.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Added patterns:      {len(result.added_patterns)}", flush=True)
    print(f"Skipped duplicates:  {len(result.skipped_duplicates)}", flush=True)
    print(f"Validation failures: {len(result.validation_failures)}", flush=True)
    print(f"Catalog path:        {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Registry path:       {paths['registry']}", flush=True)
    print(f"Index by id:         {paths['by_id']}", flush=True)
    print(f"Index by mask:       {paths['by_mask']}", flush=True)
    print(f"Index by family:     {paths['by_family']}", flush=True)

    if result.skipped_duplicates:
        print("\nSkipped duplicate generated patterns:", flush=True)
        for name in result.skipped_duplicates:
            print(f"  - {name}", flush=True)

    if result.validation_failures:
        print("\nValidation failures:", flush=True)
        for item in result.validation_failures:
            print(f"  - {item['name']} [{item['family_id']}]", flush=True)
            for error in item["errors"]:
                print(f"      * {error}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())