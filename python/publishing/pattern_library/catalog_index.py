from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from python.publishing.pattern_library.pattern_registry import PatternRegistry


def _pattern_summary(pattern) -> Dict[str, Any]:
    return {
        "pattern_id": pattern.pattern_id,
        "name": pattern.name,
        "slug": pattern.slug,
        "description": pattern.description,
        "mask81": pattern.mask81,
        "canonical_mask_signature": pattern.canonical_mask_signature,
        "clue_count": pattern.clue_count,
        "symmetry_type": pattern.symmetry_type,
        "visual_family": pattern.visual_family,
        "family_id": pattern.family_id,
        "family_name": pattern.family_name,
        "variant_code": pattern.variant_code,
        "tags": list(pattern.tags or []),
        "status": pattern.status,
        "source_type": pattern.source_type,
        "source_ref": pattern.source_ref,
        "author": pattern.author,
        "is_verified": pattern.is_verified,
        "print_score": pattern.print_score,
        "legibility_score": pattern.legibility_score,
        "aesthetic_score": pattern.aesthetic_score,
        "library_id": pattern.library_id,
    }


def build_by_id_index(registry: PatternRegistry) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for pattern in registry.patterns:
        out[str(pattern.pattern_id)] = _pattern_summary(pattern)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def build_by_mask_index(registry: PatternRegistry) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for pattern in registry.patterns:
        key = str(pattern.mask81)
        out.setdefault(key, []).append(str(pattern.pattern_id))
    for key in out:
        out[key] = sorted(out[key])
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def build_by_family_index(registry: PatternRegistry) -> Dict[str, Dict[str, Any]]:
    families: Dict[str, Dict[str, Any]] = {}
    for pattern in registry.patterns:
        family_id = str(pattern.family_id or "unclassified")
        bucket = families.setdefault(
            family_id,
            {
                "family_id": family_id,
                "family_name": pattern.family_name or family_id,
                "visual_family": pattern.visual_family or family_id,
                "pattern_ids": [],
            },
        )
        bucket["pattern_ids"].append(str(pattern.pattern_id))

    for bucket in families.values():
        bucket["pattern_ids"] = sorted(bucket["pattern_ids"])

    return dict(sorted(families.items(), key=lambda kv: kv[0]))


def write_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def build_catalog_indexes(registry: PatternRegistry, patterns_dir: Path) -> Dict[str, Path]:
    patterns_dir = Path(patterns_dir)
    indexes_dir = patterns_dir / "indexes"

    by_id_path = write_json(build_by_id_index(registry), indexes_dir / "by_id.json")
    by_mask_path = write_json(build_by_mask_index(registry), indexes_dir / "by_mask.json")
    by_family_path = write_json(build_by_family_index(registry), indexes_dir / "by_family.json")

    return {
        "by_id": by_id_path,
        "by_mask": by_mask_path,
        "by_family": by_family_path,
    }