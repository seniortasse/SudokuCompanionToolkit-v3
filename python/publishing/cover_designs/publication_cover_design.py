from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cover_design_resolver import resolve_cover_design_context, resolved_context_to_dict
from .cover_design_registry import DEFAULT_COVER_DESIGN_CATALOG
from .models import CoverDesignAssignment, ResolvedCoverDesignContext


def load_publication_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cover_design_assignment_from_publication_spec(
    publication_spec: dict[str, Any],
    publication_spec_path: str | Path | None = None,
) -> CoverDesignAssignment | None:
    block = publication_spec.get("cover_design")
    if not block:
        return None

    cover_design_id = block.get("cover_design_id")
    if not cover_design_id:
        raise ValueError("publication_spec.cover_design is missing cover_design_id")

    source = str(publication_spec_path) if publication_spec_path else "publication_spec"

    return CoverDesignAssignment(
        cover_design_id=cover_design_id,
        variables=dict(block.get("variables", {})),
        assignment_source=source,
    )


def resolve_cover_design_context_from_publication_spec(
    publication_spec_path: str | Path,
    catalog_path: str | Path = DEFAULT_COVER_DESIGN_CATALOG,
) -> ResolvedCoverDesignContext | None:
    publication_spec = load_publication_spec(publication_spec_path)
    assignment = cover_design_assignment_from_publication_spec(
        publication_spec,
        publication_spec_path=publication_spec_path,
    )

    if assignment is None:
        return None

    return resolve_cover_design_context(assignment, catalog_path=catalog_path)


def write_resolved_publication_cover_context(
    publication_spec_path: str | Path,
    out_path: str | Path,
    catalog_path: str | Path = DEFAULT_COVER_DESIGN_CATALOG,
) -> ResolvedCoverDesignContext | None:
    context = resolve_cover_design_context_from_publication_spec(
        publication_spec_path,
        catalog_path=catalog_path,
    )

    if context is None:
        return None

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(resolved_context_to_dict(context), f, indent=2, ensure_ascii=False)

    return context