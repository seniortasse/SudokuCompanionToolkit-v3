from __future__ import annotations

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas

from python.publishing.schemas.models import PageBlock, PublicationLayoutConfig
from .page_geometry import resolve_page_frame
from .render_models import PublicationRenderContext
from .typography import resolve_font_pack


_POINTS_PER_INCH = 72.0


def render_page_number(
    canvas: Canvas,
    *,
    block: PageBlock,
    context: PublicationRenderContext,
) -> None:
    if not block.show_page_number:
        return

    logical_number = block.logical_page_number
    if logical_number in (None, "", 0):
        return

    page_style = str(block.page_number_style or "arabic")
    rendered_number = _render_number(logical_number, page_style)
    if not rendered_number:
        return

    pagesize = canvas._pagesize
    mirror_margins = bool(context.publication_manifest.get("mirror_margins", False))
    physical_page_number = int(block.physical_page_number or block.page_index or 1)
    layout_config = PublicationLayoutConfig.from_dict(context.publication_manifest.get("layout_config"))
    fonts = resolve_font_pack(layout_config.font_family)

    frame = resolve_page_frame(
        page_size=pagesize,
        page_number=physical_page_number,
        mirror_margins=mirror_margins,
        inner_margin=_points(layout_config.inner_margin_in, 0.75),
        outer_margin=_points(layout_config.outer_margin_in, 0.50),
        top_margin=_points(layout_config.top_margin_in, 0.50),
        bottom_margin=_points(layout_config.bottom_margin_in, 0.50),
        trim_size=_trim_size_points(context),
        bleed=_render_bleed_points(context),
    )

    imprint_text = str(
        context.publication_manifest.get("imprint_name")
        or "Sudoku Companion"
    )

    x_left = frame.content_left
    x_right = frame.content_right
    # KDP-safe footer baseline: at least 0.42 in above the trim bottom.
    # This prevents footer labels/page numbers from being flagged as outside margins.
    y = max(frame.footer_baseline_y, frame.trim_bottom + _points(0.42, 0.42))

    canvas.setFont(fonts.regular, 10)
    canvas.setFillColor(colors.HexColor("#666666"))

    if frame.mirror_margins:
        if frame.is_even_page:
            canvas.drawString(x_left, y, rendered_number)
            canvas.drawRightString(x_right, y, imprint_text)
        else:
            canvas.drawString(x_left, y, imprint_text)
            canvas.drawRightString(x_right, y, rendered_number)
    else:
        canvas.drawString(x_left, y, imprint_text)
        canvas.drawRightString(x_right, y, rendered_number)


def _points(value_in: float | None, default_in: float) -> float:
    return float(value_in if value_in is not None else default_in) * _POINTS_PER_INCH


def _trim_size_points(context: PublicationRenderContext) -> tuple[float, float]:
    explicit = context.publication_manifest.get("_render_trim_size_pt")
    if explicit:
        return (float(explicit[0]), float(explicit[1]))

    trim = dict(context.publication_manifest.get("trim_size") or {})
    width_in = trim.get("width_in")
    height_in = trim.get("height_in")

    if width_in and height_in:
        return (float(width_in) * _POINTS_PER_INCH, float(height_in) * _POINTS_PER_INCH)

    return (float(context.publication_manifest.get("trim_width_in") or 8.5) * _POINTS_PER_INCH,
            float(context.publication_manifest.get("trim_height_in") or 11.0) * _POINTS_PER_INCH)


def _render_bleed_points(context: PublicationRenderContext) -> float:
    return float(context.publication_manifest.get("_render_bleed_pt") or 0.0)


def _render_number(value, style: str) -> str:
    number = int(value)
    if style == "arabic":
        return str(number)
    if style == "roman_lower":
        return _to_roman(number).lower()
    if style == "roman_upper":
        return _to_roman(number).upper()
    return str(number)


def _to_roman(number: int) -> str:
    if number <= 0:
        return ""

    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]

    out = []
    remaining = number
    for arabic, roman in values:
        while remaining >= arabic:
            out.append(roman)
            remaining -= arabic
    return "".join(out)