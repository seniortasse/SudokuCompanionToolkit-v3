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

    package = {}
    manifest = {}
    interior = {}
    cover_manifest = {}

    if package_path.exists():
        package = json.loads(package_path.read_text(encoding="utf-8"))

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif package.get("publication_manifest"):
        manifest = dict(package.get("publication_manifest") or {})

    if interior_path.exists():
        interior = json.loads(interior_path.read_text(encoding="utf-8"))
    elif package.get("interior_plan"):
        interior = dict(package.get("interior_plan") or {})

    if cover_manifest_path.exists():
        cover_manifest = json.loads(cover_manifest_path.read_text(encoding="utf-8"))
    elif package.get("cover_manifest"):
        cover_manifest = dict(package.get("cover_manifest") or {})

    return {
        "package": package,
        "manifest": manifest,
        "interior": interior,
        "cover_manifest": cover_manifest,
        "package_path": package_path,
        "manifest_path": manifest_path,
        "interior_path": interior_path,
        "cover_manifest_path": cover_manifest_path,
    }


def normalize_page_type(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    key = raw.lower()
    mapping = {
        "title_page": "title_page",
        "welcome_page": "welcome_page",
        "features_page": "features_page",
        "toc_page": "toc_page",
        "rules_page": "rules_page",
        "tutorial_page": "tutorial_page",
        "warmup_page": "warmup_page",
        "section_opener_page": "section_opener_page",
        "section_highlights_page": "section_highlights_page",
        "section_pattern_gallery_page": "section_pattern_gallery_page",
        "puzzle_page": "puzzle_page",
        "solution_page": "solution_page",
        "blank_page": "blank_page",
        "promo_page": "promo_page",
    }

    if key in mapping:
        return mapping[key]

    return key


def is_page_type(value: str | None, expected: str) -> bool:
    return normalize_page_type(value) == normalize_page_type(expected)