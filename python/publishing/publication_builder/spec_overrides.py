from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from python.publishing.publication_builder.layout_presets import (
    get_layout_preset,
    resolve_layout_preset,
)


def load_publication_spec_dict(spec_path: Path) -> Dict[str, Any]:
    return json.loads(spec_path.read_text(encoding="utf-8"))


def apply_publication_spec_overrides(
    spec: Dict[str, Any],
    *,
    include_solutions: Optional[bool] = None,
    page_numbering_policy: Optional[str] = None,
    layout_preset_id: Optional[str] = None,
    puzzles_per_page: Optional[int] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    gutter_x_in: Optional[float] = None,
    gutter_y_in: Optional[float] = None,
    inner_margin_in: Optional[float] = None,
    outer_margin_in: Optional[float] = None,
    top_margin_in: Optional[float] = None,
    bottom_margin_in: Optional[float] = None,
    header_height_in: Optional[float] = None,
    footer_height_in: Optional[float] = None,
    tile_slot_padding_in: Optional[float] = None,
    tile_header_band_height_in: Optional[float] = None,
    tile_gap_below_header_in: Optional[float] = None,
    tile_bottom_padding_in: Optional[float] = None,
    font_family: Optional[str] = None,
    language: Optional[str] = None,
    given_digit_size_preset: Optional[str] = None,
    solution_digit_size_preset: Optional[str] = None,
    given_digit_scale: Optional[float] = None,
    solution_digit_scale: Optional[float] = None,
) -> Dict[str, Any]:
    out = deepcopy(spec)

    if include_solutions is not None:
        out["include_solutions"] = bool(include_solutions)

    if page_numbering_policy:
        out["page_numbering_policy"] = str(page_numbering_policy)

    layout = dict(out.get("layout_config") or {})

    preset = resolve_layout_preset(
        layout_preset_id=layout_preset_id,
        puzzles_per_page=puzzles_per_page,
    )

    if preset is not None:
        out["puzzle_page_template"] = preset.puzzle_page_template
        out["solution_page_template"] = preset.solution_page_template

        layout["puzzles_per_page"] = int(preset.puzzles_per_page)
        layout["rows"] = int(preset.rows)
        layout["cols"] = int(preset.cols)
        layout["inner_margin_in"] = float(preset.inner_margin_in)
        layout["outer_margin_in"] = float(preset.outer_margin_in)
        layout["top_margin_in"] = float(preset.top_margin_in)
        layout["bottom_margin_in"] = float(preset.bottom_margin_in)
        layout["header_height_in"] = float(preset.header_height_in)
        layout["footer_height_in"] = float(preset.footer_height_in)
        layout["gutter_x_in"] = float(preset.gutter_x_in)
        layout["gutter_y_in"] = float(preset.gutter_y_in)

    if puzzles_per_page is not None:
        layout["puzzles_per_page"] = int(puzzles_per_page)

    if rows is not None:
        layout["rows"] = int(rows)

    if cols is not None:
        layout["cols"] = int(cols)

    if gutter_x_in is not None:
        layout["gutter_x_in"] = float(gutter_x_in)

    if gutter_y_in is not None:
        layout["gutter_y_in"] = float(gutter_y_in)

    if inner_margin_in is not None:
        layout["inner_margin_in"] = float(inner_margin_in)

    if outer_margin_in is not None:
        layout["outer_margin_in"] = float(outer_margin_in)

    if top_margin_in is not None:
        layout["top_margin_in"] = float(top_margin_in)

    if bottom_margin_in is not None:
        layout["bottom_margin_in"] = float(bottom_margin_in)

    if header_height_in is not None:
        layout["header_height_in"] = float(header_height_in)

    if footer_height_in is not None:
        layout["footer_height_in"] = float(footer_height_in)

    if tile_slot_padding_in is not None:
        layout["tile_slot_padding_in"] = float(tile_slot_padding_in)

    if tile_header_band_height_in is not None:
        layout["tile_header_band_height_in"] = float(tile_header_band_height_in)

    if tile_gap_below_header_in is not None:
        layout["tile_gap_below_header_in"] = float(tile_gap_below_header_in)

    if tile_bottom_padding_in is not None:
        layout["tile_bottom_padding_in"] = float(tile_bottom_padding_in)

    if font_family:
        layout["font_family"] = str(font_family)

    if language:
        layout["language"] = str(language)

    if given_digit_size_preset:
        layout["given_digit_size_preset"] = str(given_digit_size_preset)

    if solution_digit_size_preset:
        layout["solution_digit_size_preset"] = str(solution_digit_size_preset)

    if given_digit_scale is not None:
        layout["given_digit_scale"] = float(given_digit_scale)

    if solution_digit_scale is not None:
        layout["solution_digit_scale"] = float(solution_digit_scale)

    out["layout_config"] = layout

    _normalize_rows_cols_from_puzzles_per_page(out)
    _normalize_templates_from_puzzles_per_page(out)

    return out


def write_publication_spec_dict(spec: Dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _normalize_rows_cols_from_puzzles_per_page(spec: Dict[str, Any]) -> None:
    layout = dict(spec.get("layout_config") or {})
    puzzles_per_page = layout.get("puzzles_per_page")
    rows = layout.get("rows")
    cols = layout.get("cols")

    if puzzles_per_page is None:
        spec["layout_config"] = layout
        return

    count = int(puzzles_per_page)

    if rows is None and cols is None:
        preset = resolve_layout_preset(puzzles_per_page=count)
        if preset is not None:
            layout["rows"] = int(preset.rows)
            layout["cols"] = int(preset.cols)

    spec["layout_config"] = layout


def _normalize_templates_from_puzzles_per_page(spec: Dict[str, Any]) -> None:
    layout = dict(spec.get("layout_config") or {})
    puzzles_per_page = layout.get("puzzles_per_page")
    if puzzles_per_page is None:
        return

    preset = resolve_layout_preset(puzzles_per_page=int(puzzles_per_page))
    if preset is None:
        return

    spec["puzzle_page_template"] = preset.puzzle_page_template
    spec["solution_page_template"] = preset.solution_page_template