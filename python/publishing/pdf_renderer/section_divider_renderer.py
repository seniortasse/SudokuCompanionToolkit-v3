from __future__ import annotations

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas

from python.publishing.i18n.strings import tr, translate_difficulty_label
from .headline_layout import draw_smart_headline, plan_smart_headline, wrap_text_to_width_by_font
from .page_geometry import PageFrame
from .render_models import RenderSection
from .typography import resolve_font_pack


def render_section_divider_page(
    canvas: Canvas,
    section: RenderSection,
    *,
    payload: dict | None = None,
    publication_manifest: dict | None = None,
    frame: PageFrame,
    auto_advance: bool = True,
) -> None:
    payload = dict(payload or {})
    publication_manifest = dict(publication_manifest or {})

    _page_width, page_height = canvas._pagesize
    manifest = section.section_manifest
    fonts = resolve_font_pack(str((payload or {}).get("font_family") or "helvetica"))

    metadata = dict(publication_manifest.get("metadata") or {})
    layout_config = dict(publication_manifest.get("layout_config") or {})
    language = str(layout_config.get("language") or metadata.get("locale") or metadata.get("language") or "en")

    section_code = str(payload.get("section_code") or manifest.section_code)
    title = str(payload.get("title") or manifest.title)
    subtitle = str(payload.get("subtitle") or manifest.subtitle)
    puzzle_count = int(payload.get("puzzle_count") or manifest.puzzle_count or 0)
    difficulty_hint = translate_difficulty_label(
        str(payload.get("difficulty_label_hint") or manifest.difficulty_label_hint or "").strip(),
        language,
    )

    section_code_y = page_height - 120
    title_y = page_height - 155
    title_max_width = frame.content_width - 48

    title_plan = plan_smart_headline(
        text=title,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=title_max_width,
        max_lines=2,
        leading_multiplier=1.10,
    )

    title_bottom_y = title_y - ((max(title_plan["line_count"], 1) - 1) * title_plan["leading"])

    subtitle_lines = (
        wrap_text_to_width_by_font(
            subtitle,
            font_name=fonts.regular,
            font_size=13.0,
            max_width=title_max_width,
        )
        if subtitle.strip()
        else []
    )

    subtitle_y = title_bottom_y - 23
    subtitle_line_gap = 15.0
    subtitle_bottom_y = (
        subtitle_y - ((len(subtitle_lines) - 1) * subtitle_line_gap)
        if subtitle_lines
        else title_bottom_y
    )

    info_y = subtitle_bottom_y - 32
    info_bottom_y = info_y - (40 if difficulty_hint else 20)

    panel_top_y = page_height - 70
    panel_bottom_y = info_bottom_y - 18

    canvas.setFillColor(colors.HexColor("#eef3fb"))
    canvas.rect(
        frame.content_left,
        panel_bottom_y,
        frame.content_width,
        panel_top_y - panel_bottom_y,
        stroke=0,
        fill=1,
    )

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 30)
    canvas.drawString(frame.content_left + 24, section_code_y, section_code)

    draw_smart_headline(
        canvas,
        text=title,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=title_max_width,
        x=frame.content_left + 24,
        first_baseline_y=title_y,
        align="left",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    if subtitle_lines:
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 13)
        for idx, line in enumerate(subtitle_lines):
            canvas.drawString(frame.content_left + 24, subtitle_y - (idx * subtitle_line_gap), line)

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 12)
    canvas.drawString(frame.content_left + 24, info_y, f"{tr('section_order', language)}: {manifest.section_order}")
    canvas.drawString(frame.content_left + 24, info_y - 20, f"{tr('puzzles_in_section', language)}: {puzzle_count}")

    if difficulty_hint:
        canvas.drawString(frame.content_left + 24, info_y - 40, f"{tr('difficulty_hint', language)}: {difficulty_hint}")

    if auto_advance:
        canvas.showPage()