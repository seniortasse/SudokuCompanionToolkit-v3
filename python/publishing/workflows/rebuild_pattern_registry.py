from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild compiled pattern artifacts from the canonical pattern catalog/store."
    )
    parser.add_argument(
        "--patterns-dir",
        required=True,
        help="Directory containing pattern catalog / registry files.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    patterns_dir = Path(args.patterns_dir)
    registry = load_pattern_store(patterns_dir)

    rebuilt = list(registry.patterns)
    registry.patterns = rebuilt

    paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)

    print(f"Rebuilt registry:   {paths['registry']}", flush=True)
    print(f"Rebuilt catalog:    {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Rebuilt by_id:      {paths['by_id']}", flush=True)
    print(f"Rebuilt by_mask:    {paths['by_mask']}", flush=True)
    print(f"Rebuilt by_family:  {paths['by_family']}", flush=True)
    print(f"Pattern count:      {len(registry.patterns)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())