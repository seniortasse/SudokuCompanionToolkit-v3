from __future__ import annotations

import re
from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas

from python.publishing.i18n.strings import tr, translate_difficulty_label
from python.publishing.schemas.models import PuzzleRecord
from .layout_profiles import PuzzleSlot
from .typography import resolve_font_pack


@dataclass(frozen=True)
class PuzzleTileStyle:
    outer_border_color: str = "#c8cfdd"
    outer_border_width: float = 1.0
    show_slot_border: bool = False
    header_fill_color: str = "#111111"
    header_text_color: str = "#ffffff"
    header_border_color: str = "#111111"
    body_fill_color: str = "#ffffff"
    grid_line_color: str = "#111111"
    thin_grid_width: float = 0.45
    thick_grid_width: float = 1.5
    given_text_color: str = "#111111"
    solution_text_color: str = "#1f3c88"
    slot_padding: float = 10.0
    header_band_height: float = 18.0
    gap_below_header: float = 0.0
    bottom_padding: float = 10.0
    header_font_size_left: float = 9.0
    header_font_size_center: float = 9.0
    header_font_size_right: float = 9.0
    grid_digit_scale_given: float = 0.43
    grid_digit_scale_solution: float = 0.40
    font_family: str = "helvetica"
    language: str = "en"

    def with_overrides(
        self,
        *,
        slot_padding: float | None = None,
        header_band_height: float | None = None,
        gap_below_header: float | None = None,
        bottom_padding: float | None = None,
        header_font_size_left: float | None = None,
        header_font_size_center: float | None = None,
        header_font_size_right: float | None = None,
        grid_digit_scale_given: float | None = None,
        grid_digit_scale_solution: float | None = None,
        thin_grid_width: float | None = None,
        thick_grid_width: float | None = None,
        font_family: str | None = None,
        language: str | None = None,
    ) -> "PuzzleTileStyle":
        return PuzzleTileStyle(
            outer_border_color=self.outer_border_color,
            outer_border_width=self.outer_border_width,
            show_slot_border=self.show_slot_border,
            header_fill_color=self.header_fill_color,
            header_text_color=self.header_text_color,
            header_border_color=self.header_border_color,
            body_fill_color=self.body_fill_color,
            grid_line_color=self.grid_line_color,
            thin_grid_width=self.thin_grid_width if thin_grid_width is None else thin_grid_width,
            thick_grid_width=self.thick_grid_width if thick_grid_width is None else thick_grid_width,
            given_text_color=self.given_text_color,
            solution_text_color=self.solution_text_color,
            slot_padding=self.slot_padding if slot_padding is None else slot_padding,
            header_band_height=self.header_band_height if header_band_height is None else header_band_height,
            gap_below_header=self.gap_below_header if gap_below_header is None else gap_below_header,
            bottom_padding=self.bottom_padding if bottom_padding is None else bottom_padding,
            header_font_size_left=self.header_font_size_left if header_font_size_left is None else header_font_size_left,
            header_font_size_center=self.header_font_size_center if header_font_size_center is None else header_font_size_center,
            header_font_size_right=self.header_font_size_right if header_font_size_right is None else header_font_size_right,
            grid_digit_scale_given=self.grid_digit_scale_given if grid_digit_scale_given is None else grid_digit_scale_given,
            grid_digit_scale_solution=self.grid_digit_scale_solution if grid_digit_scale_solution is None else grid_digit_scale_solution,
            font_family=self.font_family if font_family is None else font_family,
            language=self.language if language is None else language,
        )


DEFAULT_PUZZLE_TILE_STYLE = PuzzleTileStyle()


def draw_puzzle_tile(
    canvas: Canvas,
    *,
    puzzle: PuzzleRecord,
    slot: PuzzleSlot,
    show_solution: bool,
    style: PuzzleTileStyle = DEFAULT_PUZZLE_TILE_STYLE,
) -> None:
    _draw_slot_background(canvas, slot, style)

    grid_x, grid_y, grid_size = _compute_grid_geometry(slot, style)

    _draw_sudoku_grid(
        canvas,
        digits=(puzzle.solution81 if show_solution else puzzle.givens81),
        x=grid_x,
        y=grid_y,
        size=grid_size,
        show_solution=show_solution,
        style=style,
    )

    _draw_header_band(
        canvas,
        puzzle=puzzle,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_size=grid_size,
        style=style,
        show_solution=show_solution,
    )


def _draw_slot_background(canvas: Canvas, slot: PuzzleSlot, style: PuzzleTileStyle) -> None:
    if not style.show_slot_border:
        return

    canvas.setFillColor(colors.white)
    canvas.setStrokeColor(colors.HexColor(style.outer_border_color))
    canvas.setLineWidth(style.outer_border_width)
    canvas.rect(slot.x, slot.y, slot.width, slot.height, stroke=1, fill=0)


def _compute_grid_geometry(slot: PuzzleSlot, style: PuzzleTileStyle) -> tuple[float, float, float]:
    available_width = slot.width - (style.slot_padding * 2)
    available_height = (
        slot.height
        - (style.slot_padding * 2)
        - style.header_band_height
    )

    grid_size = min(available_width, available_height)
    grid_x = slot.x + ((slot.width - grid_size) / 2.0)

    band_y = slot.y + slot.height - style.slot_padding - style.header_band_height
    grid_y = band_y - grid_size

    return grid_x, grid_y, grid_size


def _draw_header_band(
    canvas: Canvas,
    *,
    puzzle: PuzzleRecord,
    grid_x: float,
    grid_y: float,
    grid_size: float,
    style: PuzzleTileStyle,
    show_solution: bool,
) -> None:
    fonts = resolve_font_pack(style.font_family)

    band_x = grid_x
    band_y = grid_y + grid_size
    band_width = grid_size

    canvas.setFillColor(colors.HexColor(style.header_fill_color))
    canvas.setStrokeColor(colors.HexColor(style.header_border_color))
    canvas.setLineWidth(style.thick_grid_width)
    canvas.rect(
        band_x,
        band_y,
        band_width,
        style.header_band_height,
        stroke=1,
        fill=1,
    )

    left_label = _format_display_code(
        puzzle.print_header.display_code or puzzle.local_puzzle_code or puzzle.puzzle_uid
    )
    center_label = translate_difficulty_label(puzzle.print_header.difficulty_label, style.language) or tr("puzzle", style.language)
    right_label = tr("solution", style.language) if show_solution else _format_effort_label(puzzle.print_header.effort_label, style.language)

    left_pad = 4 if style.header_font_size_left <= 6.0 else 6
    right_pad = 4 if style.header_font_size_right <= 6.0 else 6
    text_y = band_y + max(3.0, (style.header_band_height * 0.26))

    canvas.setFillColor(colors.HexColor(style.header_text_color))

    canvas.setFont(fonts.bold, style.header_font_size_left)
    canvas.drawString(
        band_x + left_pad,
        text_y,
        left_label,
    )

    canvas.setFont(fonts.bold, style.header_font_size_center)
    canvas.drawCentredString(
        band_x + (band_width / 2.0),
        text_y,
        center_label,
    )

    canvas.setFont(fonts.bold, style.header_font_size_right)
    canvas.drawRightString(
        band_x + band_width - right_pad,
        text_y,
        right_label,
    )


def _format_display_code(value: str) -> str:
    raw = str(value or "").strip()
    match = re.match(r"^L(\d+)-(\d+)$", raw, flags=re.IGNORECASE)
    if match:
        section_num = int(match.group(1))
        ordinal_num = int(match.group(2))
        return f"L-{section_num}-{ordinal_num}"
    return raw


def _format_effort_label(value: str, language: str) -> str:
    raw = str(value or "").strip()
    match = re.match(r"^Effort\s*:?\s*(\d+)$", raw, flags=re.IGNORECASE)
    if match:
        return f"{tr('effort_prefix', language)} {match.group(1)}"
    return raw


def _draw_sudoku_grid(
    canvas: Canvas,
    *,
    digits: str,
    x: float,
    y: float,
    size: float,
    show_solution: bool,
    style: PuzzleTileStyle,
) -> None:
    fonts = resolve_font_pack(style.font_family)
    cell = size / 9.0

    canvas.setStrokeColor(colors.HexColor(style.grid_line_color))
    for i in range(10):
        line_width = style.thick_grid_width if i % 3 == 0 else style.thin_grid_width
        canvas.setLineWidth(line_width)

        vx = x + (i * cell)
        canvas.line(vx, y, vx, y + size)

        hy = y + (i * cell)
        canvas.line(x, hy, x + size, hy)

    for row in range(9):
        for col in range(9):
            value = digits[row * 9 + col]
            if value == "0":
                continue

            cx = x + (col * cell) + (cell / 2.0)
            cy = y + size - (row * cell) - (cell / 2.0)

            if show_solution:
                canvas.setFont(fonts.regular, max(8, int(cell * style.grid_digit_scale_solution)))
                canvas.setFillColor(colors.HexColor(style.solution_text_color))
            else:
                canvas.setFont(fonts.regular, max(8, int(cell * style.grid_digit_scale_given)))
                canvas.setFillColor(colors.HexColor(style.given_text_color))

            canvas.drawCentredString(cx, cy - (cell * 0.16), value)