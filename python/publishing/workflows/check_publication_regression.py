from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.workflows.publication_package_reader import (
    is_page_type,
    load_publication_artifacts,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight regression checks against a built publication package."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    parser.add_argument(
        "--expect-puzzles-per-page",
        type=int,
        default=None,
        help="Expected max puzzles per puzzle/solution page.",
    )
    parser.add_argument(
        "--expect-language",
        default=None,
        help="Expected layout_config language value.",
    )
    parser.add_argument(
        "--expect-font-family",
        default=None,
        help="Expected layout_config font_family value.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publication_dir = Path(args.publication_dir)

    _log("=" * 72)
    _log("check_publication_regression.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    try:
        artifacts = load_publication_artifacts(publication_dir)
        manifest = artifacts["manifest"]
        interior = artifacts["interior"]
    except Exception as exc:
        _log(f"ERROR: Unable to load publication artifacts: {exc}")
        return 1

    layout = dict(manifest.get("layout_config") or {})
    page_blocks = list(interior.get("page_blocks") or [])
    errors: list[str] = []

    if args.expect_language is not None:
        actual = str(layout.get("language") or "en")
        if actual != args.expect_language:
            errors.append(f"Expected language={args.expect_language}, got {actual}")

    if args.expect_font_family is not None:
        actual = str(layout.get("font_family") or "helvetica")
        if actual != args.expect_font_family:
            errors.append(f"Expected font_family={args.expect_font_family}, got {actual}")

    if args.expect_puzzles_per_page is not None:
        expected = int(args.expect_puzzles_per_page)
        for idx, block in enumerate(page_blocks, start=1):
            if not (
                is_page_type(block.get("page_type"), "puzzle_page")
                or is_page_type(block.get("page_type"), "solution_page")
            ):
                continue
            puzzle_ids = list((block.get("payload") or {}).get("puzzle_ids") or [])
            if len(puzzle_ids) > expected:
                errors.append(
                    f"Page block #{idx} has {len(puzzle_ids)} puzzles, exceeds expected max {expected}"
                )

    if errors:
        _log("REGRESSION CHECK FAILED")
        _log("-" * 72)
        for err in errors:
            _log(f"- {err}")
        _log("-" * 72)
        return 1

    _log("REGRESSION CHECK PASSED")
    _log("=" * 72)
    _log("check_publication_regression.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())