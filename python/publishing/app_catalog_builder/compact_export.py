from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from python.publishing.schemas.models import AisleManifest, CatalogManifest, PuzzleRecord
from python.publishing.techniques.technique_catalog import collapse_to_public_names


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_compact_app_catalog(
    *,
    export_dir: Path,
    catalog_manifest: CatalogManifest,
    aisle_manifests: Iterable[AisleManifest],
    puzzle_records: Iterable[PuzzleRecord],
    search_indexes: Dict[str, Dict[str, object]],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    books_dir = export_dir / "books"
    indexes_dir = export_dir / "indexes"
    puzzles_dir = export_dir / "puzzles"

    _write_json(export_dir / "catalog_manifest.json", catalog_manifest.to_dict())

    for aisle_manifest in aisle_manifests:
        filename = f"{aisle_manifest.aisle_id}.json"
        _write_json(books_dir / filename, aisle_manifest.to_dict())

    for index_name, index_payload in search_indexes.items():
        _write_json(indexes_dir / f"{index_name}.json", index_payload)

    for puzzle_record in puzzle_records:
        payload = puzzle_record.to_dict()
        payload["public_techniques_used"] = collapse_to_public_names(
            list(puzzle_record.techniques_used or []),
            plural=True,
        )
        _write_json(puzzles_dir / f"{puzzle_record.puzzle_uid}.json", payload)