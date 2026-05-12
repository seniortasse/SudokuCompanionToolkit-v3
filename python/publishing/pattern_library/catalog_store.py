from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from python.publishing.pattern_library.pattern_registry import PatternRegistry
from python.publishing.schemas.models import PatternRecord


DEFAULT_LIBRARY_ID = "LIB-CL9"
CATALOG_FILENAME = "pattern_catalog.jsonl"


def get_catalog_path(patterns_dir: Path) -> Path:
    return patterns_dir / CATALOG_FILENAME


def load_pattern_catalog(catalog_path: Path) -> PatternRegistry:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Pattern catalog not found: {catalog_path}")

    patterns: List[PatternRecord] = []
    library_id: Optional[str] = None

    with catalog_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            pattern = PatternRecord.from_dict(data)
            patterns.append(pattern)
            if library_id is None:
                library_id = pattern.library_id

    return PatternRegistry(
        library_id=library_id or DEFAULT_LIBRARY_ID,
        patterns=patterns,
    )


def save_pattern_catalog(registry: PatternRegistry, catalog_path: Path) -> None:
    registry.sort_in_place()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    with catalog_path.open("w", encoding="utf-8", newline="\n") as handle:
        for pattern in registry.patterns:
            handle.write(json.dumps(pattern.to_dict(), ensure_ascii=False))
            handle.write("\n")


def rewrite_pattern_catalog(
    *,
    patterns: Iterable[PatternRecord],
    library_id: str,
    catalog_path: Path,
) -> None:
    registry = PatternRegistry(
        library_id=library_id,
        patterns=list(patterns),
    )
    save_pattern_catalog(registry, catalog_path)