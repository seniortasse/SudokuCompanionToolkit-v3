from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CoverDesignCatalogEntry, CoverDesignRecord


DEFAULT_COVER_DESIGN_CATALOG = Path(
    "datasets/sudoku_books/classic9/cover_designs/cover_design_catalog.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cover design file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cover_design_catalog(
    catalog_path: str | Path = DEFAULT_COVER_DESIGN_CATALOG,
) -> list[CoverDesignCatalogEntry]:
    path = Path(catalog_path)
    payload = _read_json(path)

    entries: list[CoverDesignCatalogEntry] = []
    for item in payload.get("cover_designs", []):
        entries.append(
            CoverDesignCatalogEntry(
                cover_design_id=item["cover_design_id"],
                name=item["name"],
                family=item["family"],
                renderer_key=item["renderer_key"],
                status=item.get("status", "draft"),
                design_dir=item["design_dir"],
                supported_trim_sizes=list(item.get("supported_trim_sizes", [])),
                supported_channels=list(item.get("supported_channels", [])),
                default_palette_id=item.get("default_palette_id"),
                default_texture_id=item.get("default_texture_id"),
                preview_asset=item.get("preview_asset"),
            )
        )

    return entries


def find_cover_design_entry(
    cover_design_id: str,
    catalog_path: str | Path = DEFAULT_COVER_DESIGN_CATALOG,
) -> CoverDesignCatalogEntry:
    for entry in load_cover_design_catalog(catalog_path):
        if entry.cover_design_id == cover_design_id:
            return entry

    raise KeyError(f"Cover design not found in catalog: {cover_design_id}")


def load_cover_design_record(
    cover_design_id: str,
    catalog_path: str | Path = DEFAULT_COVER_DESIGN_CATALOG,
) -> CoverDesignRecord:
    catalog_file = Path(catalog_path)
    entry = find_cover_design_entry(cover_design_id, catalog_file)
    design_path = catalog_file.parent / entry.design_dir / "cover_design.json"
    payload = _read_json(design_path)

    return CoverDesignRecord(
        cover_design_id=payload["cover_design_id"],
        name=payload["name"],
        family=payload["family"],
        renderer_key=payload["renderer_key"],
        status=payload.get("status", "draft"),
        description=payload.get("description", ""),
        identity=dict(payload.get("identity", {})),
        supported_outputs=dict(payload.get("supported_outputs", {})),
        supported_trim_sizes=list(payload.get("supported_trim_sizes", [])),
        supported_channels=list(payload.get("supported_channels", [])),
        editable_variables=dict(payload.get("editable_variables", {})),
        default_variables=dict(payload.get("default_variables", {})),
        layout_regions=dict(payload.get("layout_regions", {})),
        assets=dict(payload.get("assets", {})),
    )