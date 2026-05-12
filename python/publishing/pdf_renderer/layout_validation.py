from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from python.publishing.schemas.models import PublicationLayoutConfig


_POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class LayoutValidationResult:
    ok: bool
    message: str = ""


def validate_layout_geometry(
    *,
    page_size: Tuple[float, float],
    layout_config: PublicationLayoutConfig,
    mirror_margins: bool,
) -> LayoutValidationResult:
    page_width, page_height = page_size

    rows = int(layout_config.rows or 0)
    cols = int(layout_config.cols or 0)
    puzzles_per_page = int(layout_config.puzzles_per_page or 0)

    if rows <= 0 or cols <= 0 or puzzles_per_page <= 0:
        return LayoutValidationResult(False, "rows, cols, and puzzles_per_page must all be > 0")

    if rows * cols != puzzles_per_page:
        return LayoutValidationResult(False, "rows * cols must equal puzzles_per_page")

    inner_margin = _points(layout_config.inner_margin_in, 0.75)
    outer_margin = _points(layout_config.outer_margin_in, 0.50)
    top_margin = _points(layout_config.top_margin_in, 0.50)
    bottom_margin = _points(layout_config.bottom_margin_in, 0.50)
    header_height = _points(layout_config.header_height_in, 0.33)
    footer_height = _points(layout_config.footer_height_in, 0.33)
    gutter_x = _points(layout_config.gutter_x_in, 0.25)
    gutter_y = _points(layout_config.gutter_y_in, 0.25)

    tile_slot_padding = _points(layout_config.tile_slot_padding_in, 10.0 / 72.0)
    tile_header_band_height = _points(layout_config.tile_header_band_height_in, 18.0 / 72.0)
    tile_gap_below_header = _points(layout_config.tile_gap_below_header_in, 0.0)
    tile_bottom_padding = _points(layout_config.tile_bottom_padding_in, 10.0 / 72.0)

    if min(
        inner_margin,
        outer_margin,
        top_margin,
        bottom_margin,
        header_height,
        footer_height,
        gutter_x,
        gutter_y,
        tile_slot_padding,
        tile_header_band_height,
        tile_gap_below_header,
        tile_bottom_padding,
    ) < 0:
        return LayoutValidationResult(False, "layout values must all be >= 0")

    content_width = page_width - inner_margin - outer_margin
    content_height = page_height - top_margin - bottom_margin

    if content_width <= 0 or content_height <= 0:
        return LayoutValidationResult(False, "margins leave no usable page content area")

    usable_width = content_width - ((cols - 1) * gutter_x)
    usable_height = content_height - header_height - footer_height - ((rows - 1) * gutter_y)

    if usable_width <= 0 or usable_height <= 0:
        return LayoutValidationResult(False, "gutters/header/footer leave no usable slot area")

    slot_width = usable_width / float(cols)
    slot_height = usable_height / float(rows)

    if slot_width <= 0 or slot_height <= 0:
        return LayoutValidationResult(False, "slot width or slot height became non-positive")

    available_grid_width = slot_width - (tile_slot_padding * 2)
    available_grid_height = (
        slot_height
        - (tile_slot_padding * 2)
        - tile_header_band_height
        - tile_gap_below_header
        - tile_bottom_padding
    )

    if available_grid_width <= 0 or available_grid_height <= 0:
        return LayoutValidationResult(
            False,
            "tile padding/header settings leave no room for the grid inside a slot",
        )

    grid_size = min(available_grid_width, available_grid_height)
    if grid_size < 72:
        return LayoutValidationResult(
            False,
            f"grid_size is too small ({grid_size:.1f}pt); reduce puzzles_per_page or spacing",
        )

    return LayoutValidationResult(True, "")


def _points(value_in: float | None, default_in: float) -> float:
    return float(value_in if value_in is not None else default_in) * _POINTS_PER_INCH