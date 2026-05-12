from __future__ import annotations

from typing import List

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas

from python.publishing.schemas.models import PublicationLayoutConfig, PuzzleRecord
from .headline_layout import draw_smart_headline
from .layout_profiles import LayoutProfile
from .puzzle_tile_renderer import (
    DEFAULT_PUZZLE_TILE_STYLE,
    PuzzleTileStyle,
    draw_puzzle_tile,
)
from .typography import resolve_digit_scale, resolve_font_pack


_POINTS_PER_INCH = 72.0


def render_puzzle_pages(
    canvas: Canvas,
    *,
    puzzles: List[PuzzleRecord],
    layout_profile: LayoutProfile,
    page_title: str,
    show_solution: bool,
    layout_config: PublicationLayoutConfig | None = None,
) -> None:
    slot_count = layout_profile.puzzles_per_page
    for start in range(0, len(puzzles), slot_count):
        chunk = puzzles[start : start + slot_count]
        render_puzzle_page(
            canvas,
            puzzles=chunk,
            layout_profile=layout_profile,
            page_title=page_title,
            show_solution=show_solution,
            layout_config=layout_config,
        )
        canvas.showPage()


def render_puzzle_page(
    canvas: Canvas,
    *,
    puzzles: List[PuzzleRecord],
    layout_profile: LayoutProfile,
    page_title: str,
    show_solution: bool,
    layout_config: PublicationLayoutConfig | None = None,
) -> None:
    canvas.setFillColor(colors.HexColor("#ffffff"))

    _draw_page_header(
        canvas,
        page_title=page_title,
        layout_profile=layout_profile,
        layout_config=layout_config,
    )

    tile_style = _tile_style_for_layout(
        layout_profile,
        show_solution=show_solution,
        layout_config=layout_config,
    )

    for puzzle, slot in zip(puzzles, layout_profile.slots):
        draw_puzzle_tile(
            canvas,
            puzzle=puzzle,
            slot=slot,
            show_solution=show_solution,
            style=tile_style,
        )


def _draw_page_header(
    canvas: Canvas,
    *,
    page_title: str,
    layout_profile: LayoutProfile,
    layout_config: PublicationLayoutConfig | None = None,
) -> None:
    cfg = layout_config or PublicationLayoutConfig()
    fonts = resolve_font_pack(cfg.font_family)

    frame = layout_profile.frame

    header_on_outer_right = bool(frame.mirror_margins and not frame.is_even_page)
    header_x = frame.content_right if header_on_outer_right else frame.content_left
    header_align = "right" if header_on_outer_right else "left"

    title_block = draw_smart_headline(
        canvas,
        text=page_title,
        font_name=fonts.bold,
        preferred_font_size=12.0,
        min_font_size=8.5,
        max_width=frame.content_width,
        x=header_x,
        first_baseline_y=frame.trim_top - 30,
        align=header_align,
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.HexColor("#1f3c88"),
    )

    underline_y = title_block["bottom_y"] - 6

    canvas.setStrokeColor(colors.HexColor("#d9dfeb"))
    canvas.setLineWidth(0.8)
    canvas.line(
        frame.content_left,
        underline_y,
        frame.content_right,
        underline_y,
    )


def _tile_style_for_layout(
    layout_profile: LayoutProfile,
    *,
    show_solution: bool,
    layout_config: PublicationLayoutConfig | None = None,
) -> PuzzleTileStyle:
    cfg = layout_config or PublicationLayoutConfig()

    if layout_profile.puzzles_per_page <= 2:
        base = PuzzleTileStyle(
            slot_padding=12.0,
            header_band_height=20.0,
            gap_below_header=0.0,
            bottom_padding=12.0,
            header_font_size_left=10.0,
            header_font_size_center=9.0,
            header_font_size_right=9.0,
            grid_digit_scale_given=0.44,
            grid_digit_scale_solution=0.41,
        )
    elif layout_profile.puzzles_per_page <= 4:
        base = PuzzleTileStyle(
            gap_below_header=0.0,
            header_font_size_center=8.5,
        )
    elif layout_profile.puzzles_per_page <= 6:
        base = PuzzleTileStyle(
            slot_padding=8.0,
            header_band_height=15.0,
            gap_below_header=0.0,
            bottom_padding=8.0,
            header_font_size_left=7.8,
            header_font_size_center=7.8,
            header_font_size_right=7.8,
            grid_digit_scale_given=0.38,
            grid_digit_scale_solution=0.36,
            thin_grid_width=0.40,
            thick_grid_width=1.35,
        )
    else:
        base = PuzzleTileStyle(
            slot_padding=5.0,
            header_band_height=11.0,
            gap_below_header=0.0,
            bottom_padding=5.0,
            header_font_size_left=5.8,
            header_font_size_center=5.4,
            header_font_size_right=5.4,
            grid_digit_scale_given=0.30,
            grid_digit_scale_solution=0.28,
            thin_grid_width=0.32,
            thick_grid_width=1.05,
        )

    return base.with_overrides(
        slot_padding=_points_or_none(cfg.tile_slot_padding_in),
        header_band_height=_points_or_none(cfg.tile_header_band_height_in),
        gap_below_header=_points_or_none(cfg.tile_gap_below_header_in),
        bottom_padding=_points_or_none(cfg.tile_bottom_padding_in),
        grid_digit_scale_given=resolve_digit_scale(
            explicit_scale=cfg.given_digit_scale,
            preset=cfg.given_digit_size_preset,
            fallback=base.grid_digit_scale_given,
        ),
        grid_digit_scale_solution=resolve_digit_scale(
            explicit_scale=cfg.solution_digit_scale,
            preset=cfg.solution_digit_size_preset,
            fallback=base.grid_digit_scale_solution,
        ),
        font_family=str(cfg.font_family or "helvetica"),
        language=str(cfg.language or "en"),
    )


def _points_or_none(value_in: float | None) -> float | None:
    if value_in is None:
        return None
    return float(value_in) * _POINTS_PER_INCH