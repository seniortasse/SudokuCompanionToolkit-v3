from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.pattern_library.catalog_index import build_catalog_indexes
from python.publishing.pattern_library.pattern_store import load_pattern_store


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compiled pattern catalog indexes from the canonical pattern store."
    )
    parser.add_argument(
        "--patterns-dir",
        required=True,
        help="Directory containing pattern_catalog.jsonl / registry.json",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    patterns_dir = Path(args.patterns_dir)
    registry = load_pattern_store(patterns_dir)
    paths = build_catalog_indexes(registry, patterns_dir)

    print(f"Patterns dir: {patterns_dir}", flush=True)
    print(f"Pattern count: {len(registry.patterns)}", flush=True)
    print(f"Index by id:   {paths['by_id']}", flush=True)
    print(f"Index by mask: {paths['by_mask']}", flush=True)
    print(f"Index by family: {paths['by_family']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())