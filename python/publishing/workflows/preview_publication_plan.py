from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from python.publishing.workflows.publication_package_reader import (
    load_publication_artifacts,
    normalize_page_type,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the structure of a built publication package."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum number of page blocks to preview.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publication_dir = Path(args.publication_dir)

    _log("=" * 72)
    _log("preview_publication_plan.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    try:
        artifacts = load_publication_artifacts(publication_dir)
    except Exception as exc:
        _log(f"ERROR: Unable to load publication artifacts: {exc}")
        return 1

    publication_manifest = artifacts["manifest"]
    interior_plan = artifacts["interior"]
    cover_manifest = artifacts["cover_manifest"]

    if not publication_manifest:
        _log("ERROR: publication_manifest.json is missing or empty")
        return 1
    if not interior_plan:
        _log("ERROR: interior_plan.json is missing or empty")
        return 1

    blocks = interior_plan.get("page_blocks", []) or []
    counter = Counter(normalize_page_type(block.get("page_type", "UNKNOWN")) for block in blocks)

    _log(f"Title:                {publication_manifest.get('book_title', '-')}")
    _log(f"Publication ID:       {publication_manifest.get('publication_id', '-')}")
    _log(f"Book ID:              {publication_manifest.get('book_id', '-')}")
    _log(f"Puzzle template:      {publication_manifest.get('puzzle_page_template', '-')}")
    _log(f"Solution template:    {publication_manifest.get('solution_page_template', '-')}")
    _log(f"Page numbering:       {publication_manifest.get('page_numbering_policy', '-')}")
    _log(f"Format:               {publication_manifest.get('format_id', '-')}")
    _log(f"Trim:                 {publication_manifest.get('trim_size', {})}")
    _log(f"Mirror margins:       {publication_manifest.get('mirror_margins', False)}")
    _log(f"Estimated pages:      {interior_plan.get('estimated_page_count', '-')}")
    _log(f"Blank-page adjusted:  {interior_plan.get('requires_blank_page_adjustment', False)}")
    _log(f"Front matter profile: {publication_manifest.get('front_matter_profile', '-')}")
    _log(f"End matter profile:   {publication_manifest.get('end_matter_profile', '-')}")
    _log("-" * 72)

    layout = dict(publication_manifest.get("layout_config") or {})
    if layout:
        _log("Layout config:")
        for key in sorted(layout.keys()):
            _log(f"  {key:<28} {layout[key]}")
        _log("-" * 72)

    _log("Page-type counts:")
    for key in sorted(counter.keys()):
        _log(f"  {key:<24} {counter[key]}")

    if cover_manifest:
        geometry = cover_manifest.get("geometry", {})
        _log("-" * 72)
        _log("Cover:")
        _log(f"  template:           {cover_manifest.get('cover_template', '-')}")
        _log(f"  total width (in):   {geometry.get('total_width_in', '-')}")
        _log(f"  total height (in):  {geometry.get('total_height_in', '-')}")
        _log(f"  spine width (in):   {geometry.get('spine_width_in', '-')}")
        _log(f"  spine text:         {cover_manifest.get('spine_text', '-')}")

    _log("-" * 72)
    _log(f"First {min(args.limit, len(blocks))} page blocks:")
    for block in blocks[: args.limit]:
        page_index = block.get("page_index")
        physical_page_number = block.get("physical_page_number", "")
        logical_page_number = block.get("logical_page_number", "")
        page_type = normalize_page_type(block.get("page_type"))
        template_id = block.get("template_id")
        show_num = block.get("show_page_number", False)
        puzzle_count = len(list((block.get("payload") or {}).get("puzzle_ids") or []))
        _log(
            f"  idx={page_index:>3} | phys={physical_page_number:>3} | "
            f"logical={logical_page_number:>3} | {page_type:<22} | "
            f"{template_id:<28} | show_page_number={show_num:<5} | puzzles={puzzle_count}"
        )

    _log("=" * 72)
    _log("preview_publication_plan.py completed")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())