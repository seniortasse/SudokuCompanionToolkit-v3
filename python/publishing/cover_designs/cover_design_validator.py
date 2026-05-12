from __future__ import annotations

from typing import Any

from .models import ResolvedCoverDesignContext


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload

    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current


def validate_resolved_cover_design_context(
    context: ResolvedCoverDesignContext,
) -> list[str]:
    errors: list[str] = []

    if not context.cover_design_id:
        errors.append("Missing cover_design_id.")

    if not context.renderer_key:
        errors.append("Missing renderer_key.")

    required_top_level = ["text", "palette_id", "texture_id", "puzzle_art", "features"]
    for key in required_top_level:
        if key not in context.variables:
            errors.append(f"Missing required variable group: {key}")

    required_text_fields = [
        "text.year",
        "text.puzzle_count_label",
        "text.title_word",
        "text.difficulty_label",
    ]
    for dotted_key in required_text_fields:
        value = _get_nested(context.variables, dotted_key)
        if value in (None, ""):
            errors.append(f"Missing required text variable: {dotted_key}")

    

    required_puzzle_sources = [
        "puzzle_art.main_grid_source.mode",
        "puzzle_art.left_side_grid_source.mode",
        "puzzle_art.right_side_grid_source.mode",
    ]
    for dotted_key in required_puzzle_sources:
        value = _get_nested(context.variables, dotted_key)
        if value in (None, ""):
            errors.append(f"Missing required puzzle-art variable: {dotted_key}")

    palette = context.variables.get("palette", {})
    if palette is not None and not isinstance(palette, dict):
        errors.append("palette must be an object when provided.")

    features = context.variables.get("features", {})
    if features is not None and not isinstance(features, dict):
        errors.append("features must be an object when provided.")

    return errors