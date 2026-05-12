from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.app_catalog_builder.android_asset_export import export_catalog_to_android_assets


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the compact app catalog export into Android app assets."
    )
    parser.add_argument(
        "--source-export-dir",
        default="exports/sudoku_books/app_catalog/classic9",
        help="Path to the compact app catalog export directory.",
    )
    parser.add_argument(
        "--android-assets-dir",
        default="python/mobile/android/app/src/main/assets/sudoku_library/classic9",
        help="Destination directory inside Android assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    source_export_dir = Path(args.source_export_dir)
    android_assets_dir = Path(args.android_assets_dir)

    _log("=" * 72)
    _log("export_app_catalog.py starting")
    _log("=" * 72)
    _log(f"Source export dir:    {source_export_dir.resolve()}")
    _log(f"Android assets dir:   {android_assets_dir.resolve()}")
    _log("=" * 72)

    if not source_export_dir.exists():
        _log(f"ERROR: source export dir not found: {source_export_dir}")
        return 1

    try:
        written_dir = export_catalog_to_android_assets(
            source_export_dir=source_export_dir,
            android_assets_dir=android_assets_dir,
        )
    except Exception as exc:
        _log(f"ERROR while exporting to Android assets: {exc}")
        return 1

    _log(f"Android asset catalog written to: {written_dir}")
    _log("=" * 72)
    _log("export_app_catalog.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())