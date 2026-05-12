from __future__ import annotations

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from python.publishing.i18n.strings import tr
from .page_geometry import PageFrame
from .render_models import BuiltBookRenderModel
from .typography import resolve_font_pack


def render_title_page(
    canvas: Canvas,
    render_model: BuiltBookRenderModel,
    *,
    frame: PageFrame,
    payload: dict | None = None,
    publication_manifest: dict | None = None,
    auto_advance: bool = True,
) -> None:
    payload = dict(payload or {})
    publication_manifest = dict(publication_manifest or {})

    page_width, page_height = canvas._pagesize
    book = render_model.book_manifest
    fonts = resolve_font_pack(payload.get("font_family") or "helvetica")

    metadata = dict(publication_manifest.get("metadata") or {})
    layout_config = dict(publication_manifest.get("layout_config") or {})
    language = str(layout_config.get("language") or metadata.get("locale") or metadata.get("language") or "en")

    title = str(payload.get("title") or publication_manifest.get("book_title") or book.title)
    subtitle = str(payload.get("subtitle") or publication_manifest.get("book_subtitle") or book.subtitle)
    #series_name = str(payload.get("series_name") or publication_manifest.get("series_name") or book.series_name)
    description = str(
        payload.get("description")
        or metadata.get("description")
        or book.description
    ).strip()

    trim_size = str(publication_manifest.get("trim_size_label") or book.trim_size)
    puzzles_per_page = int((layout_config.get("puzzles_per_page") or book.puzzles_per_page or 0))
    total_puzzles = int(book.puzzle_count or 0)
    book_code = str(payload.get("book_code") or metadata.get("book_code") or "").strip()

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - 96, page_width, 96, stroke=0, fill=1)

    canvas.setFillColor(colors.white)
    canvas.setFont(fonts.bold, 22)
    canvas.drawString(frame.content_left, page_height - 58, title)

    if subtitle.strip():
        canvas.setFont(fonts.regular, 12)
        canvas.drawString(frame.content_left, page_height - 80, subtitle)

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 12)

    y = page_height - 160

    # KDP-facing title page:
    # Keep reader/commercial facts.
    # Hide internal production/debug fields such as series id, volume id,
    # library id, aisle id, and layout type.
    lines = [
        f"{tr('trim_size', language)}: {trim_size}",
        f"{tr('puzzles_per_page', language)}: {puzzles_per_page}",
        f"{tr('total_puzzles', language)}: {total_puzzles}",
    ]

    if book_code:
        lines.append(f"{tr('book_code', language)}: {book_code}")

    for line in lines:
        canvas.drawString(frame.content_left, y, line)
        y -= 20

    if description:
        y -= 8
        canvas.setFont(fonts.bold, 12)
        canvas.drawString(frame.content_left, y, tr("description", language))
        y -= 20
        canvas.setFont(fonts.regular, 11)
        text = canvas.beginText(frame.content_left, y)
        text.setLeading(15)
        for paragraph_line in _wrap_text(description, width=88):
            text.textLine(paragraph_line)
        canvas.drawText(text)

    footer_line = str(metadata.get("generated_by_pipeline") or tr("generated_by_pipeline", language)).strip()
    if footer_line:
        canvas.setFont(fonts.italic, 10)
        canvas.setFillColor(colors.HexColor("#555555"))

        # Keep the generated-by line safely above the global footer/page-number line.
        #
        # In bleed mode, the global footer is drawn higher because trim_bottom is
        # shifted up by the bleed strip.  A raw y=42 collides with the footer.
        # This hard floor guarantees the title-page generated-by line sits clearly
        # above the footer in both bleed and no-bleed interiors.
        footer_line_y = max(frame.footer_baseline_y + 15.0, frame.trim_bottom + 50.0, 60.0)
        canvas.drawString(frame.content_left, footer_line_y, footer_line)

        if auto_advance:
            canvas.showPage()


def _wrap_text(value: str, width: int) -> list[str]:
    words = value.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines