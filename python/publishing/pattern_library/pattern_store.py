from __future__ import annotations

from pathlib import Path
from typing import Dict

from python.publishing.pattern_library.catalog_index import build_catalog_indexes
from python.publishing.pattern_library.catalog_store import get_catalog_path, load_pattern_catalog, save_pattern_catalog
from python.publishing.pattern_library.pattern_registry import PatternRegistry, load_registry, save_registry


DEFAULT_LIBRARY_ID = "LIB-CL9"


def load_pattern_store(patterns_dir: Path) -> PatternRegistry:
    patterns_dir = Path(patterns_dir)
    catalog_path = get_catalog_path(patterns_dir)
    registry_path = patterns_dir / "registry.json"

    if catalog_path.exists():
        return load_pattern_catalog(catalog_path)

    if registry_path.exists():
        return load_registry(registry_path)

    return PatternRegistry(library_id=DEFAULT_LIBRARY_ID)


def save_pattern_store(registry: PatternRegistry, patterns_dir: Path) -> Path:
    patterns_dir = Path(patterns_dir)
    catalog_path = get_catalog_path(patterns_dir)
    registry_path = patterns_dir / "registry.json"

    save_pattern_catalog(registry, catalog_path)
    save_registry(registry, registry_path)
    return registry_path


def rebuild_compiled_pattern_artifacts(registry: PatternRegistry, patterns_dir: Path) -> Dict[str, Path]:
    patterns_dir = Path(patterns_dir)
    registry_path = save_pattern_store(registry, patterns_dir)
    index_paths = build_catalog_indexes(registry, patterns_dir)

    out: Dict[str, Path] = {"registry": registry_path}
    out.update(index_paths)
    return out