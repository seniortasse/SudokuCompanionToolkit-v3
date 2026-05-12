from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from python.publishing.schemas.models import (
    CoverSpec,
    InteriorPlan,
    PrintFormatSpec,
    PublicationPackage,
    PublicationSpec,
)


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_publication_spec(path: Path) -> PublicationSpec:
    if not path.exists():
        raise FileNotFoundError(f"Publication spec not found: {path}")
    return PublicationSpec.from_dict(_read_json(path))


def load_print_format_spec(path: Path) -> PrintFormatSpec:
    if not path.exists():
        raise FileNotFoundError(f"Print format spec not found: {path}")
    return PrintFormatSpec.from_dict(_read_json(path))


def load_interior_plan(path: Path) -> InteriorPlan:
    if not path.exists():
        raise FileNotFoundError(f"Interior plan not found: {path}")
    return InteriorPlan.from_dict(_read_json(path))


def load_cover_spec(path: Path) -> CoverSpec:
    if not path.exists():
        raise FileNotFoundError(f"Cover spec not found: {path}")
    return CoverSpec.from_dict(_read_json(path))


def load_publication_package(path: Path) -> PublicationPackage:
    if not path.exists():
        raise FileNotFoundError(f"Publication package manifest not found: {path}")
    return PublicationPackage.from_dict(_read_json(path))