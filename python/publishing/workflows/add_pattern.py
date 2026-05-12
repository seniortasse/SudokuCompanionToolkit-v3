from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from python.publishing.pattern_library.pattern_enricher import enrich_pattern_record
from python.publishing.pattern_library.pattern_identity import build_variant_code
from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)
from python.publishing.pattern_library.pattern_validator import validate_pattern_record_strict
from python.publishing.schemas.models import PatternRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a pattern from a JSON payload file.")
    parser.add_argument("--patterns-dir", default="datasets/sudoku_books/classic9/patterns")
    parser.add_argument("--payload-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    patterns_dir = Path(args.patterns_dir)
    payload_path = Path(args.payload_json)

    registry = load_pattern_store(patterns_dir)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    if not registry.library_id:
        registry.library_id = str(payload.get("library_id") or "LIB-CL9")

    library_short = str(registry.library_id).split("-")[-1]
    next_ordinal = registry.get_next_ordinal()
    timestamp = _now_iso()

    pattern = PatternRecord(
        pattern_id=payload.get("pattern_id") or f"PAT-{library_short}-{next_ordinal:04d}",
        library_id=registry.library_id,
        name=str(payload["name"]),
        slug=str(payload.get("slug", "")),
        aliases=list(payload.get("aliases", [])),
        description=str(payload.get("description", "")),
        grid_size=int(payload.get("grid_size", 9)),
        layout_type=str(payload.get("layout_type", "classic9x9")),
        mask81=str(payload["mask81"]),
        canonical_mask_signature=str(payload.get("canonical_mask_signature", "")),
        clue_count=int(payload.get("clue_count", sum(1 for ch in str(payload["mask81"]) if ch == "1"))),
        symmetry_type=str(payload.get("symmetry_type", "none")),
        visual_family=str(payload.get("visual_family", "uncategorized")),
        family_id=payload.get("family_id"),
        family_name=payload.get("family_name"),
        variant_code=payload.get("variant_code") or build_variant_code(family_name=payload.get("family_name"), ordinal=1),
        tags=list(payload.get("tags", [])),
        status=str(payload.get("status", "active")),
        source_type=str(payload.get("source_type", "manual")),
        source_ref=payload.get("source_ref"),
        author=payload.get("author"),
        notes=payload.get("notes"),
        is_verified=bool(payload.get("is_verified", False)),
        print_score=payload.get("print_score"),
        legibility_score=payload.get("legibility_score"),
        aesthetic_score=payload.get("aesthetic_score"),
        created_at=payload.get("created_at") or timestamp,
        updated_at=payload.get("updated_at") or timestamp,
    )

    pattern = enrich_pattern_record(pattern)
    errors = validate_pattern_record_strict(pattern)
    if errors:
        print(f"Pattern validation failed for {pattern.name}", flush=True)
        for error in errors:
            print(f"  * {error}", flush=True)
        return 1

    was_added = registry.add_or_skip_duplicate_mask(pattern)
    if not was_added:
        print(
            "Pattern not added because another pattern with the same canonical mask identity already exists.",
            flush=True,
        )
        return 1

    paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)
    print(f"Added pattern:   {pattern.pattern_id}", flush=True)
    print(f"Registry path:   {paths['registry']}", flush=True)
    print(f"Catalog path:    {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Index by id:     {paths['by_id']}", flush=True)
    print(f"Index by mask:   {paths['by_mask']}", flush=True)
    print(f"Index by family: {paths['by_family']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())