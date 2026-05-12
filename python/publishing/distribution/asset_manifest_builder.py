from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def build_asset_manifest(*, bundle_dir: Path) -> Dict[str, object]:
    assets: List[Dict[str, object]] = []

    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(bundle_dir).as_posix()
            assets.append(
                {
                    "path": rel,
                    "category": _classify_asset(rel),
                    "size_bytes": path.stat().st_size,
                }
            )

    return {
        "bundle_dir": str(bundle_dir),
        "asset_count": len(assets),
        "assets": assets,
    }


def _classify_asset(rel_path: str) -> str:
    path = rel_path.lower()

    if path.endswith(".pdf"):
        if "cover" in path:
            return "pdf_cover"
        if "interior" in path:
            return "pdf_interior"
        return "pdf_other"

    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return "preview_image"

    if path.endswith("metadata.json") or path.endswith("kdp_profile.json"):
        return "distribution_metadata"

    if path.endswith("asset_manifest.json") or path.endswith("bundle_summary.json"):
        return "bundle_manifest"

    if path.endswith(".json"):
        return "publication_manifest"

    return "other"