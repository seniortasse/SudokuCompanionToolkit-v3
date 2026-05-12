from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive a pattern by pattern_id.")
    parser.add_argument("--patterns-dir", default="datasets/sudoku_books/classic9/patterns")
    parser.add_argument("--pattern-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    patterns_dir = Path(args.patterns_dir)

    registry = load_pattern_store(patterns_dir)
    ok = registry.archive_pattern(args.pattern_id)
    if not ok:
        print(f"Pattern not found: {args.pattern_id}", flush=True)
        return 1

    paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)
    print(f"Archived pattern: {args.pattern_id}", flush=True)
    print(f"Registry updated: {paths['registry']}", flush=True)
    print(f"Catalog updated:  {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Index by id:      {paths['by_id']}", flush=True)
    print(f"Index by mask:    {paths['by_mask']}", flush=True)
    print(f"Index by family:  {paths['by_family']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())