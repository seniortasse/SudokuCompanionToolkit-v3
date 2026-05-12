from __future__ import annotations

import shutil
from pathlib import Path


def export_catalog_to_android_assets(
    *,
    source_export_dir: Path,
    android_assets_dir: Path,
) -> Path:
    """
    Copy the compact app catalog bundle into Android assets.

    Example:
        source_export_dir = exports/sudoku_books/app_catalog/classic9
        android_assets_dir = python/mobile/android/app/src/main/assets/sudoku_library/classic9
    """
    if not source_export_dir.exists():
        raise FileNotFoundError(f"Source export dir not found: {source_export_dir}")

    android_assets_dir.parent.mkdir(parents=True, exist_ok=True)

    if android_assets_dir.exists():
        shutil.rmtree(android_assets_dir)

    shutil.copytree(source_export_dir, android_assets_dir)
    return android_assets_dir