from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .cover_design_registry import (
    DEFAULT_COVER_DESIGN_CATALOG,
    find_cover_design_entry,
    load_cover_design_record,
)
from .models import CoverDesignAssignment, ResolvedCoverDesignContext


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_cover_design_assignment(path: str | Path) -> CoverDesignAssignment:
    assignment_path = Path(path)

    with assignment_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cover_design_id = payload.get("cover_design_id")
    if not cover_design_id:
        raise ValueError(f"Missing cover_design_id in assignment: {assignment_path}")

    return CoverDesignAssignment(
        cover_design_id=cover_design_id,
        variables=dict(payload.get("variables", {})),
        assignment_source=str(assignment_path),
    )


def resolve_cover_design_context(
    assignment: CoverDesignAssignment,
    catalog_path: str | Path = DEFAULT_COVER_DESIGN_CATALOG,
) -> ResolvedCoverDesignContext:
    catalog_file = Path(catalog_path)
    entry = find_cover_design_entry(assignment.cover_design_id, catalog_file)
    record = load_cover_design_record(assignment.cover_design_id, catalog_file)

    variables = deep_merge(record.default_variables, assignment.variables)
    design_dir = catalog_file.parent / entry.design_dir

    return ResolvedCoverDesignContext(
        cover_design_id=record.cover_design_id,
        name=record.name,
        family=record.family,
        renderer_key=record.renderer_key,
        status=record.status,
        variables=variables,
        editable_variables=record.editable_variables,
        identity=record.identity,
        layout_regions=record.layout_regions,
        assets=record.assets,
        design_dir=design_dir,
        catalog_path=catalog_file,
        assignment_source=assignment.assignment_source,
    )


def resolved_context_to_dict(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return {
        "cover_design_id": context.cover_design_id,
        "name": context.name,
        "family": context.family,
        "renderer_key": context.renderer_key,
        "status": context.status,
        "assignment_source": context.assignment_source,
        "catalog_path": str(context.catalog_path),
        "design_dir": str(context.design_dir),
        "identity": context.identity,
        "layout_regions": context.layout_regions,
        "assets": context.assets,
        "editable_variables": context.editable_variables,
        "variables": context.variables,
    }