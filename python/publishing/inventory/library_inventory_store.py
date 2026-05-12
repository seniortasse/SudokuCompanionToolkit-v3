from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

INVENTORY_FILENAME = "_library_inventory.json"


def _default_inventory(*, library_id: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "library_id": library_id,
        "records": {},
    }


def _inventory_path(base_dir: Path) -> Path:
    return base_dir / INVENTORY_FILENAME


def load_library_inventory(*, base_dir: Path, library_id: str) -> Dict[str, Any]:
    path = _inventory_path(base_dir)
    if not path.exists():
        return _default_inventory(library_id=library_id)

    data = json.loads(path.read_text(encoding="utf-8"))
    inventory = _default_inventory(library_id=library_id)
    inventory.update(data)
    inventory["records"] = dict(inventory.get("records", {}))

    if str(inventory.get("library_id", library_id)) != library_id:
        raise ValueError(
            f"Inventory library_id mismatch: expected {library_id}, found {inventory.get('library_id')}"
        )

    return inventory


def save_library_inventory(*, inventory: Dict[str, Any], base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = _inventory_path(base_dir)
    path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path