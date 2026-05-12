from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_publication_artifacts(publication_dir: Path) -> Dict[str, Any]:
    publication_dir = Path(publication_dir)

    package_path = publication_dir / "publication_package.json"
    manifest_path = publication_dir / "publication_manifest.json"
    interior_path = publication_dir / "interior_plan.json"
    cover_manifest_path = publication_dir / "cover_manifest.json"

    if not package_path.exists():
        raise FileNotFoundError(f"publication_package.json not found in {publication_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"publication_manifest.json not found in {publication_dir}")
    if not interior_path.exists():
        raise FileNotFoundError(f"interior_plan.json not found in {publication_dir}")

    package = json.loads(package_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interior = json.loads(interior_path.read_text(encoding="utf-8"))

    cover_manifest: Dict[str, Any] = {}
    if cover_manifest_path.exists():
        cover_manifest = json.loads(cover_manifest_path.read_text(encoding="utf-8"))

    return {
        "publication_package": package,
        "publication_manifest": manifest,
        "interior_plan": interior,
        "cover_manifest": cover_manifest,
    }