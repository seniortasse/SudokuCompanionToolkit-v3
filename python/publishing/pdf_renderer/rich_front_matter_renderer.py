from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from python.publishing.i18n.strings import format_of_total, tr
from python.publishing.techniques.technique_catalog import get_public_technique_name
from .headline_layout import draw_smart_headline
from .page_geometry import PageFrame
from .typography import resolve_font_pack


# KDP-safe blue-header label baseline.
# On bleed interiors, page_height includes the bleed strip, so header labels
# must be positioned from frame.trim_top, not from the PDF MediaBox top.
# 54 pt = 0.75 in below the trim top, safely beyond KDP's 0.375 in top text margin.
_BLUE_HEADER_LABEL_FROM_TRIM_TOP_PT = 54.0

# Bottom front-matter notes live in the narrow band between the last content
# block and the global footer/page-number line.
#
# IMPORTANT:
# ReportLab y coordinates increase upward.
#   - Larger value = text moves UP.
#   - Smaller value = text moves DOWN.
#
# This is an absolute baseline in PDF points for the FIRST line of the final
# italic note on Rules / Tutorial / Warm-up pages.
#
# Good tuning range:
#   62.0 = lower, closer to footer
#   68.0 = recommended starting point
#   74.0 = higher, closer to content block above
_FRONT_MATTER_BOTTOM_NOTE_BASELINE_Y_PT = 68.0

# Features-page bottom availability row. Same coordinate rule:
# larger = up, smaller = down.
_FRONT_MATTER_AUX_LINE_BASELINE_Y_PT = 66.0


def _blue_header_label_y(frame: PageFrame) -> float:
    return frame.trim_top - _BLUE_HEADER_LABEL_FROM_TRIM_TOP_PT


def _draw_blue_header_label(canvas: Canvas, frame: PageFrame, text: str) -> None:
    if frame.mirror_margins and not frame.is_even_page:
        canvas.drawRightString(frame.content_right, _blue_header_label_y(frame), str(text))
    else:
        canvas.drawString(frame.content_left, _blue_header_label_y(frame), str(text))


def _front_matter_bottom_note_y(frame: PageFrame) -> float:
    return _FRONT_MATTER_BOTTOM_NOTE_BASELINE_Y_PT


def _front_matter_aux_line_y(frame: PageFrame) -> float:
    return _FRONT_MATTER_AUX_LINE_BASELINE_Y_PT


def _get_editorial_copy(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    editorial = dict(payload.get("editorial_copy") or {})
    return dict(editorial.get(key) or {})


def _payload_language(payload: Dict[str, Any]) -> str:
    return str(payload.get("language") or "en")


def _localized_series_header(eyebrow: str, index: int, total: int, language: str) -> str:
    if int(total) > 1:
        return f"{eyebrow} • {format_of_total(index, total, language)}"
    return eyebrow


def _wrap_text(value: str, width: int) -> list[str]:
    words = str(value or "").split()
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


def _draw_soft_panel(canvas: Canvas, *, x: float, y: float, w: float, h: float, fill_hex: str) -> None:
    canvas.setFillColor(colors.HexColor(fill_hex))
    canvas.setStrokeColor(colors.HexColor(fill_hex))
    canvas.rect(x, y, w, h, stroke=0, fill=1)


def _wrap_text_to_width(value: str, *, font_name: str, font_size: float, max_width: float) -> list[str]:
    words = str(value or "").split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_paragraph_block(
    canvas: Canvas,
    *,
    x: float,
    y_top: float,
    width: float,
    paragraphs: Sequence[str],
    font_name: str,
    font_size: float,
    leading: float,
    paragraph_gap: float,
    color=colors.black,
) -> float:
    y = y_top
    canvas.setFillColor(color)
    canvas.setFont(font_name, font_size)

    for paragraph in paragraphs:
        lines = _wrap_text_to_width(
            str(paragraph),
            font_name=font_name,
            font_size=font_size,
            max_width=width,
        )
        for line in lines:
            canvas.drawString(x, y, line)
            y -= leading
        y -= paragraph_gap

    return y


def _resolve_logo_path(payload: Dict[str, Any], brand_copy: Dict[str, Any]) -> Path | None:
    candidates = [
        str(brand_copy.get("logo_path") or "").strip(),
        str(payload.get("logo_path") or "").strip(),
        "logo/love_your_brain.png",
    ]
    project_root = Path(__file__).resolve().parents[3]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
        rooted = project_root / candidate
        if rooted.is_file():
            return rooted
    return None


def _resolve_asset_path(value: str) -> Path | None:
    candidate = str(value or "").strip()
    if not candidate:
        return None

    path = Path(candidate)
    if path.is_file():
        return path

    project_root = Path(__file__).resolve().parents[3]
    rooted = project_root / candidate
    if rooted.is_file():
        return rooted

    return None


def _extract_book_code(book_id: str) -> str:
    raw = str(book_id or "").strip().upper()
    if not raw:
        return ""

    parts = [part.strip() for part in raw.split("-") if part.strip()]
    for part in reversed(parts):
        if re.fullmatch(r"B\d{2,}", part):
            return part

    match = re.search(r"(B\d{2,})$", raw)
    if match:
        return match.group(1)

    return ""


def _normalize_solution_language_suffix(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""

    match = re.match(r"([a-z]{2})", raw)
    if match:
        return match.group(1)

    sanitized = "".join(ch for ch in raw if ch.isalpha())
    return sanitized[:2]


def _build_solution_support_url(payload: Dict[str, Any], fallback_url: str = "") -> str:
    book_id = str(payload.get("book_id") or "").strip()
    language = str(
        payload.get("language")
        or payload.get("publication_language")
        or ""
    ).strip()

    book_code = _extract_book_code(book_id)
    lang_suffix = _normalize_solution_language_suffix(language)

    if book_code and lang_suffix:
        return f"www.contextionary.com/solution{book_code}{lang_suffix}"

    return str(fallback_url or "").strip()


def _normalize_layout_suffix(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""

    match = re.search(r"(\d+)", raw)
    if match:
        return f"{int(match.group(1))}up"

    return ""


def _build_review_url(payload: Dict[str, Any], fallback_url: str = "") -> str:
    book_id = str(payload.get("book_id") or "").strip()
    language = str(
        payload.get("language")
        or payload.get("publication_language")
        or ""
    ).strip()
    puzzles_per_page = payload.get("publication_puzzles_per_page")

    book_code = _extract_book_code(book_id)
    lang_suffix = _normalize_solution_language_suffix(language)
    layout_suffix = _normalize_layout_suffix(puzzles_per_page)

    if book_code and lang_suffix and layout_suffix:
        return f"www.contextionary.com/review{book_code}{lang_suffix}{layout_suffix}"

    return str(fallback_url or "").strip()


def _draw_label_and_link(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    label: str,
    link_text: str,
    fonts,
    label_font_size: float = 9.6,
    link_font_size: float = 9.6,
) -> None:
    label = str(label or "").strip()
    link_text = str(link_text or "").strip()
    if not label and not link_text:
        return

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, label_font_size)
    canvas.drawString(x, y, label)

    label_w = pdfmetrics.stringWidth(label, fonts.regular, label_font_size)
    link_x = x + label_w + 4

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.regular, link_font_size)
    canvas.drawString(link_x, y, link_text)

    link_w = pdfmetrics.stringWidth(link_text, fonts.regular, link_font_size)
    canvas.setLineWidth(0.8)
    canvas.setStrokeColor(colors.HexColor("#1f3c88"))
    canvas.line(link_x, y - 1.5, link_x + link_w, y - 1.5)


def _draw_story_feature_card(
    canvas: Canvas,
    *,
    x: float,
    y_top: float,
    w: float,
    h: float,
    fonts,
    eyebrow: str,
    title: str,
    body: str,
    takeaway: str,
    accent_hex: str = "#1f3c88",
) -> None:
    _draw_soft_panel(canvas, x=x, y=y_top - h, w=w, h=h, fill_hex="#f6f7fb")

    canvas.setFillColor(colors.HexColor(accent_hex))
    canvas.rect(x, y_top - h, 6, h, stroke=0, fill=1)

    inner_x = x + 18
    inner_w = w - 32
    yy = y_top - 18

    if eyebrow:
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 10.0)
        canvas.drawString(inner_x, yy, eyebrow)
        yy -= 16

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.bold, 14.0)
    for line in _wrap_text_to_width(
        title,
        font_name=fonts.bold,
        font_size=14.0,
        max_width=inner_w,
    ):
        canvas.drawString(inner_x, yy, line)
        yy -= 16

    yy -= 2
    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 10.0)
    for line in _wrap_text_to_width(
        body,
        font_name=fonts.regular,
        font_size=10.0,
        max_width=inner_w,
    ):
        canvas.drawString(inner_x, yy, line)
        yy -= 13

    if takeaway:
        yy -= 3
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.setFont(fonts.italic, 9.6)
        for line in _wrap_text_to_width(
            takeaway,
            font_name=fonts.italic,
            font_size=9.6,
            max_width=inner_w,
        ):
            canvas.drawString(inner_x, yy, line)
            yy -= 12


def _draw_title_block(
    canvas: Canvas,
    *,
    frame: PageFrame,
    fonts,
    eyebrow: str = "",
    title: str,
    subtitle: str = "",
    body_lines: Sequence[str] = (),
) -> float:
    page_height = frame.page_height
    x = frame.content_left
    y_top = page_height - 42

    if eyebrow:
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.setFont(fonts.bold, 11)
        canvas.drawString(x, y_top, eyebrow)

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.bold, 24)
    canvas.drawString(x, y_top - 30, title)

    current_y = y_top - 58

    if subtitle:
        canvas.setFillColor(colors.HexColor("#333333"))
        canvas.setFont(fonts.regular, 12.5)
        for line in _wrap_text(subtitle, width=95):
            canvas.drawString(x, current_y, line)
            current_y -= 15
        current_y -= 8

    if body_lines:
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 11)
        for line in body_lines:
            canvas.drawString(x, current_y, str(line))
            current_y -= 16

    return current_y


def _draw_feature_cards(
    canvas: Canvas,
    *,
    frame: PageFrame,
    fonts,
    items: Sequence[Dict[str, str]],
    start_y: float,
) -> float:
    card_gap = 12
    card_w = (frame.content_width - card_gap) / 2.0
    card_h = 92
    x1 = frame.content_left
    x2 = frame.content_left + card_w + card_gap
    y = start_y

    for idx, item in enumerate(items):
        x = x1 if idx % 2 == 0 else x2
        if idx > 0 and idx % 2 == 0:
            y -= card_h + 12

        _draw_soft_panel(canvas, x=x, y=y - card_h, w=card_w, h=card_h, fill_hex="#f4f6fa")

        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 11.5)
        canvas.drawString(x + 12, y - 18, str(item.get("title") or "").strip())

        body = str(item.get("body") or "").strip()
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.8)

        yy = y - 36
        for line in _wrap_text(body, width=40):
            canvas.drawString(x + 12, yy, line)
            yy -= 13

    if items:
        rows = ((len(items) - 1) // 2) + 1
        return start_y - rows * (card_h + 12)
    return start_y


def render_welcome_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    page_width = frame.page_width
    page_height = frame.page_height
    fonts = resolve_font_pack(payload.get("font_family"))
    welcome_copy = _get_editorial_copy(payload, "welcome")
    brand_copy = _get_editorial_copy(payload, "brand")
    ecosystem_cfg = dict(payload.get("ecosystem_config") or {})
    language = _payload_language(payload)

    logo_path = _resolve_logo_path(payload, brand_copy)
    support_email = str(ecosystem_cfg.get("support_email") or "").strip()
    subscribe_url = str(ecosystem_cfg.get("subscribe_url") or "").strip()

    headline = str(welcome_copy.get("headline") or payload.get("title") or tr("welcome", language))
    kicker = str(
        welcome_copy.get("kicker")
        or brand_copy.get("identity_line")
        or "Beauty for the eye. Logic for the mind. A friendly guide for the journey."
    ).strip()

    paragraphs = list(welcome_copy.get("body_paragraphs") or [])
    if not paragraphs:
        fallback = str(
            welcome_copy.get("body")
            or payload.get("description")
            or "Welcome to your Sudoku Companion volume. This book was assembled to support a satisfying, progressive solving journey."
        ).strip()
        if fallback:
            paragraphs = [fallback]

    closing = str(welcome_copy.get("closing") or "").strip()
    support_label = str(welcome_copy.get("support_label") or "Share a note with us").strip()
    subscribe_label = str(welcome_copy.get("subscribe_label") or "Stay in touch").strip()

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    header_h = 104
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, page_width, header_h, stroke=0, fill=1)

    content_left = frame.content_left
    content_width = frame.content_width
    center_x = content_left + (content_width / 2.0)

    band_label = str(
        welcome_copy.get("eyebrow")
        or payload.get("band_label")
        or tr("welcome", language)
    ).strip()

    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    _draw_blue_header_label(canvas, frame, band_label)

    logo_top = page_height - 26
    if logo_path is not None:
        try:
            img = ImageReader(str(logo_path))
            iw, ih = img.getSize()
            target_w = min(content_width * 0.44, 180)
            target_h = target_w * (float(ih) / float(iw or 1))
            draw_x = center_x - (target_w / 2.0)
            draw_y = logo_top - target_h
            canvas.drawImage(img, draw_x, draw_y, width=target_w, height=target_h, mask="auto")
            logo_bottom = draw_y
        except Exception:
            logo_bottom = page_height - 112
    else:
        logo_bottom = page_height - 112

    headline_block = draw_smart_headline(
        canvas,
        text=headline,
        font_name=fonts.bold,
        preferred_font_size=20.5,
        min_font_size=15.5,
        max_width=content_width * 0.92,
        x=center_x,
        first_baseline_y=logo_bottom - 16,
        align="center",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    kicker_y = headline_block["bottom_y"] - 20
    kicker_lines = _wrap_text_to_width(
        kicker,
        font_name=fonts.italic,
        font_size=10.8,
        max_width=content_width * 0.90,
    )
    canvas.setFillColor(colors.HexColor("#4e4e4e"))
    canvas.setFont(fonts.italic, 10.8)
    yy = kicker_y
    for line in kicker_lines:
        canvas.drawCentredString(center_x, yy, line)
        yy -= 13

    panel_top = yy - 10
    footer_start_y = 82
    panel_bottom = footer_start_y + 38
    panel_height = max(280, panel_top - panel_bottom)

    _draw_soft_panel(
        canvas,
        x=content_left,
        y=panel_bottom,
        w=content_width,
        h=panel_height,
        fill_hex="#f6f7fb",
    )

    text_x = content_left + 18
    text_width = content_width - 36
    text_top = panel_bottom + panel_height - 24

    _draw_paragraph_block(
        canvas,
        x=text_x,
        y_top=text_top,
        width=text_width,
        paragraphs=paragraphs,
        font_name=fonts.regular,
        font_size=10.15,
        leading=13.2,
        paragraph_gap=6.0,
        color=colors.black,
    )

    footer_y = footer_start_y + 20

    if closing:
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.setFont(fonts.regular, 9.6)
        closing_lines = _wrap_text_to_width(
            closing,
            font_name=fonts.regular,
            font_size=9.6,
            max_width=text_width,
        )
        for line in closing_lines[:2]:
            canvas.drawString(content_left, footer_y, line)
            footer_y -= 11
        footer_y -= 3

    if support_email:
        _draw_label_and_link(
            canvas,
            x=content_left,
            y=footer_y,
            label=f"{support_label}:",
            link_text=support_email,
            fonts=fonts,
            label_font_size=9.6,
            link_font_size=9.6,
        )
        footer_y -= 13

    if subscribe_url:
        _draw_label_and_link(
            canvas,
            x=content_left,
            y=footer_y,
            label=f"{subscribe_label}:",
            link_text=subscribe_url,
            fonts=fonts,
            label_font_size=9.6,
            link_font_size=9.6,
        )


def render_features_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    page_width = frame.page_width
    page_height = frame.page_height
    fonts = resolve_font_pack(payload.get("font_family"))
    features_copy = _get_editorial_copy(payload, "features")
    features_cfg = dict(payload.get("features_page_config") or {})
    ecosystem_cfg = dict(payload.get("ecosystem_config") or {})
    publication_metadata = dict(payload.get("publication_metadata") or {})
    language = _payload_language(payload)

    raw_audience = str(publication_metadata.get("audience") or "Adults").strip()
    audience = tr("audience_adults", language) if raw_audience.lower() == "adults" else raw_audience
    title = str(features_copy.get("headline") or "The page is only the beginning").strip()
    kicker = str(
        features_copy.get("kicker")
        or "A beautiful printed book, with help when you want it."
    ).strip()
    intro_body = str(
        features_copy.get("body")
        or "This collection is designed to feel good in your hands, clear on the page, and generous when you want a little help."
    ).strip()
    footer_cta = str(features_copy.get("footer_cta") or "").strip()

    cards = list(features_cfg.get("cards") or features_copy.get("cards") or [])
    bottom_note_title = str(features_cfg.get("bottom_note_title") or features_copy.get("bottom_note_title") or "").strip()
    bottom_note_body = str(features_cfg.get("bottom_note_body") or features_copy.get("bottom_note_body") or "").strip()

    solution_cfg = dict(ecosystem_cfg.get("qr_solution_support") or {})
    companion_cfg = dict(ecosystem_cfg.get("companion_app") or {})
    future_cfg = dict(ecosystem_cfg.get("future_tools") or {})

    available_now_label = str(
        features_cfg.get("available_now_label") or tr("available_now_label", language)
    ).strip()
    coming_soon_label = str(
        features_cfg.get("coming_soon_label") or tr("coming_soon_label", language)
    ).strip()

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    header_h = 92
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, page_width, header_h, stroke=0, fill=1)

    content_left = frame.content_left
    content_width = frame.content_width
    center_x = content_left + (content_width / 2.0)

    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    companion_features = tr("companion_features", language)
    if audience:
        _draw_blue_header_label(canvas, frame, f"{audience} • {companion_features}")
    else:
        _draw_blue_header_label(canvas, frame, companion_features)

    title_block = draw_smart_headline(
        canvas,
        text=title,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=content_width * 0.92,
        x=center_x,
        first_baseline_y=page_height - 118,
        align="center",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    kicker_y = title_block["bottom_y"] - 20
    canvas.setFillColor(colors.HexColor("#4e4e4e"))
    canvas.setFont(fonts.italic, 10.8)
    kicker_lines = _wrap_text_to_width(
        kicker,
        font_name=fonts.italic,
        font_size=10.8,
        max_width=content_width * 0.9,
    )
    for idx, line in enumerate(kicker_lines):
        canvas.drawCentredString(center_x, kicker_y - (idx * 13), line)

    intro_top = kicker_y - ((len(kicker_lines) - 1) * 13 if kicker_lines else 0) - 26
    intro_h = 78
    _draw_soft_panel(canvas, x=content_left, y=intro_top - intro_h, w=content_width, h=intro_h, fill_hex="#f7f8fb")
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(content_left, intro_top - intro_h, 5, intro_h, stroke=0, fill=1)

    intro_text_x = content_left + 18
    intro_text_w = content_width - 36
    intro_y = intro_top - 20
    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 10.0)
    for line in _wrap_text_to_width(
        intro_body,
        font_name=fonts.regular,
        font_size=10.0,
        max_width=intro_text_w,
    ):
        canvas.drawString(intro_text_x, intro_y, line)
        intro_y -= 12

    if footer_cta:
        intro_y -= 2
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 9.5)
        for line in _wrap_text_to_width(
            footer_cta,
            font_name=fonts.italic,
            font_size=9.5,
            max_width=intro_text_w,
        ):
            canvas.drawString(intro_text_x, intro_y, line)
            intro_y -= 11

    cards_top = intro_top - intro_h - 16
    card_h = 100
    card_gap = 12
    card_x = content_left
    card_w = content_width
    card_items = cards[:3]

    for idx, card in enumerate(card_items):
        y_top = cards_top - idx * (card_h + card_gap)
        _draw_story_feature_card(
            canvas,
            x=card_x,
            y_top=y_top,
            w=card_w,
            h=card_h,
            fonts=fonts,
            eyebrow=str(card.get("eyebrow") or "").strip(),
            title=str(card.get("title") or "").strip(),
            body=str(card.get("body") or "").strip(),
            takeaway=str(card.get("takeaway") or "").strip(),
        )

    bottom_y = 78
    bottom_h = 82
    _draw_soft_panel(
        canvas,
        x=content_left,
        y=bottom_y,
        w=content_width,
        h=bottom_h,
        fill_hex="#eef3fb",
    )
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(content_left, bottom_y, 5, bottom_h, stroke=0, fill=1)

    note_x = content_left + 18
    note_y = bottom_y + bottom_h - 18

    if bottom_note_title:
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 10.8)
        for line in _wrap_text_to_width(
            bottom_note_title,
            font_name=fonts.bold,
            font_size=10.8,
            max_width=content_width - 36,
        ):
            canvas.drawString(note_x, note_y, line)
            note_y -= 12
        note_y -= 2

    if bottom_note_body:
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.5)
        for line in _wrap_text_to_width(
            bottom_note_body,
            font_name=fonts.regular,
            font_size=9.5,
            max_width=content_width - 36,
        )[:4]:
            canvas.drawString(note_x, note_y, line)
            note_y -= 11

    availability_y = _front_matter_aux_line_y(frame)
    if bool(solution_cfg.get("enabled", False)):
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 9.4)
        canvas.drawString(content_left, availability_y, available_now_label)
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.4)
        canvas.drawString(content_left + 70, availability_y, str(solution_cfg.get("label") or "Solution App"))

    if bool(companion_cfg.get("enabled", False)) or bool(future_cfg.get("enabled", False)):
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 9.4)
        canvas.drawString(content_left + 228, availability_y, coming_soon_label)
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.4)
        canvas.drawString(content_left + 304, availability_y, str(companion_cfg.get("label") or "Sudoku Companion App"))


def render_toc_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    page_height = frame.page_height
    language = str(payload.get("language") or "en")
    fonts = resolve_font_pack(payload.get("font_family"))

    _draw_soft_panel(
        canvas,
        x=frame.content_left,
        y=page_height - 116,
        w=frame.content_width,
        h=74,
        fill_hex="#eef3fb",
    )

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 22)
    canvas.drawString(frame.content_left + 18, page_height - 74, tr("contents", language))

    entries = payload.get("entries", []) or []

    y = page_height - 152
    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 11)

    for entry in entries:
        title = str(entry.get("title", "")).strip()
        page_index = entry.get("page_index")
        if not title or y < 72:
            continue
        canvas.drawString(frame.content_left + 16, y, title)
        canvas.drawRightString(frame.content_right - 16, y, str(page_index))
        y -= 18


def _format_display_puzzle_code(value: str) -> str:
    raw = str(value or "").strip()
    match = re.match(r"^L(\d+)-(\d+)$", raw, flags=re.IGNORECASE)
    if match:
        section_num = int(match.group(1))
        ordinal_num = int(match.group(2))
        return f"L-{section_num}-{ordinal_num}"
    return raw


def _format_frontmatter_puzzle_reference_line(value: str, language: str = "en") -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw

    known_prefixes = [
        tr("source_puzzle", language),
        tr("practice_puzzle", language),
        "Source puzzle:",
        "Practice puzzle:",
        "Puzzle source :",
        "Puzzle d’entraînement :",
    ]

    for prefix in known_prefixes:
        if raw.startswith(prefix):
            remainder = raw[len(prefix):].strip()
            parts = [part.strip() for part in remainder.split("•")]
            if parts:
                parts[0] = _format_display_puzzle_code(parts[0])
                return f"{prefix} {' • '.join(parts)}"
            return raw
    return raw


def _parse_cell_ref(value: Any):
    if isinstance(value, dict):
        r = int(value.get("r", 0))
        c = int(value.get("c", 0))
        if 1 <= r <= 9 and 1 <= c <= 9:
            return r, c
        return None

    s = str(value or "").strip().lower()
    if s.startswith("r") and "c" in s:
        try:
            r_part, c_part = s[1:].split("c", 1)
            r = int(r_part)
            c = int(c_part)
            if 1 <= r <= 9 and 1 <= c <= 9:
                return r, c
        except Exception:
            return None
    return None


def _cell_rect_from_rc(x: float, y_top: float, cell: float, r: int, c: int):
    left = x + (c - 1) * cell
    top = y_top - (r - 1) * cell
    return left, top - cell, cell, cell


def _draw_sudoku_board(
    canvas: Canvas,
    *,
    x: float,
    y_top: float,
    size: float,
    grid81: str,
    fonts=None,
    highlight_rows: Sequence[int] = (),
    highlight_cols: Sequence[int] = (),
    highlight_boxes: Sequence[int] = (),
    target_cells: Sequence[Any] = (),
    witness_cells: Sequence[Any] = (),
    candidate_cells: Sequence[Any] = (),
    pattern_cells: Sequence[Any] = (),
    cell_candidates: Dict[str, Any] | None = None,
    pattern_locked_digits: Dict[str, Any] | None = None,
    result_cell: Any = None,
    result_digit: Any = None,
) -> None:
    grid81 = str(grid81 or "")
    if len(grid81) != 81:
        raise ValueError(f"_draw_sudoku_board expected grid81 length 81, got {len(grid81)}")

    cell_candidates = {
        str(k).strip().lower(): v
        for k, v in dict(cell_candidates or {}).items()
    }

    pattern_locked_digits = {
        str(k).strip().lower(): v
        for k, v in dict(pattern_locked_digits or {}).items()
    }

    cell = size / 9.0

    canvas.setFillColor(colors.white)
    canvas.rect(x, y_top - size, size, size, stroke=0, fill=1)

    for row_idx in highlight_rows:
        if 1 <= int(row_idx) <= 9:
            yy = y_top - int(row_idx) * cell
            canvas.setFillColor(colors.HexColor("#eaf1ff"))
            canvas.rect(x, yy, size, cell, stroke=0, fill=1)

    for col_idx in highlight_cols:
        if 1 <= int(col_idx) <= 9:
            xx = x + (int(col_idx) - 1) * cell
            canvas.setFillColor(colors.HexColor("#eef5ff"))
            canvas.rect(xx, y_top - size, cell, size, stroke=0, fill=1)

    for box_idx in highlight_boxes:
        if 1 <= int(box_idx) <= 9:
            br = (int(box_idx) - 1) // 3
            bc = (int(box_idx) - 1) % 3
            xx = x + bc * 3 * cell
            yy = y_top - (br * 3 + 3) * cell
            canvas.setFillColor(colors.HexColor("#fff4e6"))
            canvas.rect(xx, yy, 3 * cell, 3 * cell, stroke=0, fill=1)

    for ref in candidate_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setFillColor(colors.HexColor("#f4f6fa"))
        canvas.rect(xx, yy, ww, hh, stroke=0, fill=1)

    for ref in pattern_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setFillColor(colors.HexColor("#e7f6ec"))
        canvas.rect(xx, yy, ww, hh, stroke=0, fill=1)

    for ref in witness_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setFillColor(colors.HexColor("#fff1d6"))
        canvas.rect(xx, yy, ww, hh, stroke=0, fill=1)

    for ref in target_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setFillColor(colors.HexColor("#dfe7fb"))
        canvas.rect(xx, yy, ww, hh, stroke=0, fill=1)

    for i in range(10):
        lw = 1.4 if i % 3 == 0 else 0.45
        canvas.setLineWidth(lw)
        canvas.setStrokeColor(colors.black)
        canvas.line(x + i * cell, y_top, x + i * cell, y_top - size)
        canvas.line(x, y_top - i * cell, x + size, y_top - i * cell)

    for r in range(1, 10):
        for c in range(1, 10):
            ref = f"r{r}c{c}"
            ch = grid81[(r - 1) * 9 + (c - 1)]
            xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)

            if ch not in (".", "0"):
                canvas.setFillColor(colors.black)
                canvas.setFont((fonts.bold if fonts else "Helvetica-Bold"), max(8, cell * 0.48))
                canvas.drawCentredString(xx + ww / 2.0, yy + hh * 0.28, ch)
                continue

            locked_digit = pattern_locked_digits.get(ref)

            # If this cell has a locked/circled digit overlay, draw that instead of
            # the small candidate list. This is important for cumulative intersection
            # stories, where a prior pointing pair may be shown as orange witness cells.
            if locked_digit is not None:
                cx = xx + ww / 2.0
                cy = yy + hh * 0.30
                radius = max(5.0, cell * 0.16)

                canvas.setStrokeColor(colors.HexColor("#1f3c88"))
                canvas.setLineWidth(1.0)
                canvas.circle(cx, cy, radius, stroke=1, fill=0)

                canvas.setFillColor(colors.HexColor("#1f3c88"))
                canvas.setFont((fonts.bold if fonts else "Helvetica-Bold"), max(6.5, cell * 0.22))
                canvas.drawCentredString(cx, cy - (radius * 0.35), str(locked_digit))
            else:
                cand_value = cell_candidates.get(ref)
                if cand_value:
                    if isinstance(cand_value, (list, tuple)):
                        cand_text = " ".join(str(v) for v in cand_value)
                    else:
                        cand_text = str(cand_value)

                    normalized_pattern_cells = {str(x).strip().lower() for x in pattern_cells}
                    canvas.setFillColor(
                        colors.HexColor("#2f855a")
                        if ref in normalized_pattern_cells
                        else colors.HexColor("#555555")
                    )
                    canvas.setFont((fonts.regular if fonts else "Helvetica"), max(5.5, cell * 0.17))
                    canvas.drawCentredString(xx + ww / 2.0, yy + hh * 0.52, cand_text)

    for ref in pattern_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setLineWidth(1.5)
        canvas.setStrokeColor(colors.HexColor("#2f855a"))
        canvas.rect(xx + 1.3, yy + 1.3, ww - 2.6, hh - 2.6, stroke=1, fill=0)

    for ref in witness_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setLineWidth(1.3)
        canvas.setStrokeColor(colors.HexColor("#d98a00"))
        canvas.rect(xx + 1.3, yy + 1.3, ww - 2.6, hh - 2.6, stroke=1, fill=0)

    for ref in target_cells:
        rc = _parse_cell_ref(ref)
        if rc is None:
            continue
        r, c = rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setLineWidth(1.8)
        canvas.setStrokeColor(colors.HexColor("#1f3c88"))
        canvas.rect(xx + 1.2, yy + 1.2, ww - 2.4, hh - 2.4, stroke=1, fill=0)

    result_rc = _parse_cell_ref(result_cell)
    if result_rc is not None and str(result_digit or "").strip():
        r, c = result_rc
        xx, yy, ww, hh = _cell_rect_from_rc(x, y_top, cell, r, c)
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold if fonts else "Helvetica-Bold", max(9, cell * 0.56))
        canvas.drawCentredString(xx + ww / 2.0, yy + hh * 0.28, str(result_digit))


def _draw_candidate_zoom(
    canvas: Canvas,
    *,
    x: float,
    y_top: float,
    w: float,
    h: float,
    digits: Sequence[int],
    kept_digit: int,
    crossed_digits: Sequence[int],
    fonts,
) -> None:
    _draw_soft_panel(canvas, x=x, y=y_top - h, w=w, h=h, fill_hex="#f7f8fb")
    canvas.setStrokeColor(colors.HexColor("#cfd6e5"))
    canvas.setLineWidth(0.8)
    canvas.rect(x, y_top - h, w, h, stroke=1, fill=0)

    cols = max(1, len(digits))
    cell_w = w / cols

    for idx, digit in enumerate(digits):
        cx = x + idx * cell_w
        canvas.setLineWidth(0.5)
        canvas.setStrokeColor(colors.HexColor("#d7dbe6"))
        canvas.rect(cx, y_top - h, cell_w, h, stroke=1, fill=0)

        color = colors.HexColor("#1f3c88") if int(digit) == int(kept_digit) else colors.black
        canvas.setFillColor(color)
        canvas.setFont(fonts.bold, 16)
        canvas.drawCentredString(cx + cell_w / 2.0, y_top - h / 2.0 - 5, str(digit))

        if int(digit) in {int(x) for x in crossed_digits}:
            canvas.setStrokeColor(colors.HexColor("#c1121f"))
            canvas.setLineWidth(1.6)
            canvas.line(cx + 7, y_top - 7, cx + cell_w - 7, y_top - h + 7)
            canvas.line(cx + 7, y_top - h + 7, cx + cell_w - 7, y_top - 7)


def _friendly_rule_label(value: str, language: str = "en") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    known = {
        "HOUSE_CANDIDATE_CELLS_FOR_DIGIT": tr("hidden_single", language),
        "CELL_CANDIDATE_DIGITS": tr("naked_single", language),
    }
    if raw in known:
        return known[raw]

    key = raw.lower().replace("-", "_").replace(" ", "_")
    localized = {
        "singles_1": tr("hidden_single", language),
        "singles_2": tr("hidden_single", language),
        "singles_3": tr("hidden_single", language),
        "singles_naked_2": tr("naked_single", language),
        "singles_naked_3": tr("naked_single", language),
        "doubles_naked": tr("naked_pair", language),
        "triplets_naked": tr("naked_triple", language),
        "singles_pointing": tr("pointing_pair", language),
        "singles_boxed": tr("claiming_pair", language),
    }
    if key in localized:
        return localized[key]

    return get_public_technique_name(raw, plural=False)


def _draw_rules_example_panel(
    canvas: Canvas,
    *,
    x: float,
    y_top: float,
    w: float,
    h: float,
    fonts,
    title: str,
    formal: str,
    body: str,
    facts: Sequence[str],
    board_kwargs: Dict[str, Any],
    candidate_zoom: Dict[str, Any] | None = None,
) -> None:
    _draw_soft_panel(canvas, x=x, y=y_top - h, w=w, h=h, fill_hex="#f7f8fb")

    board_size = 158
    board_x = x + 16
    board_y_top = y_top - 18

    _draw_sudoku_board(
        canvas,
        x=board_x,
        y_top=board_y_top,
        size=board_size,
        grid81=str(board_kwargs.get("grid81") or ""),
        fonts=fonts,
        highlight_rows=board_kwargs.get("highlight_rows") or [],
        highlight_cols=board_kwargs.get("highlight_cols") or [],
        highlight_boxes=board_kwargs.get("highlight_boxes") or [],
        target_cells=board_kwargs.get("target_cells") or [],
        witness_cells=board_kwargs.get("witness_cells") or [],
        candidate_cells=board_kwargs.get("candidate_cells") or [],
        pattern_cells=board_kwargs.get("pattern_cells") or [],
        cell_candidates=board_kwargs.get("cell_candidates") or {},
        pattern_locked_digits=board_kwargs.get("pattern_locked_digits") or {},
        result_cell=board_kwargs.get("result_cell"),
        result_digit=board_kwargs.get("result_digit"),
    )

    text_x = board_x + board_size + 18
    text_w = w - (text_x - x) - 16
    yy = y_top - 18

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 10.0)
    panel_label = _friendly_rule_label(str(formal or "").strip(), board_kwargs.get("language") or "en")
    if panel_label:
        canvas.drawString(text_x, yy, panel_label)
        yy -= 16

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.bold, 14.2)
    for line in _wrap_text_to_width(title, font_name=fonts.bold, font_size=14.2, max_width=text_w):
        canvas.drawString(text_x, yy, line)
        yy -= 16

    yy -= 1
    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 9.9)
    for line in _wrap_text_to_width(body, font_name=fonts.regular, font_size=9.9, max_width=text_w):
        canvas.drawString(text_x, yy, line)
        yy -= 12

    yy -= 3
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.setFont(fonts.regular, 9.3)
    for fact in facts:
        wrapped = _wrap_text_to_width(f"• {fact}", font_name=fonts.regular, font_size=9.3, max_width=text_w)
        for line in wrapped:
            canvas.drawString(text_x, yy, line)
            yy -= 11


def render_rules_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    fonts = resolve_font_pack(payload.get("font_family"))
    rules_copy = _get_editorial_copy(payload, "rules")
    page_index = int(payload.get("rules_page_index") or payload.get("page_occurrence_index") or 1)
    page_total = int(payload.get("rules_page_total") or payload.get("page_occurrence_total") or 1)
    language = _payload_language(payload)

    page1 = dict(rules_copy.get("page1") or {})
    page2 = dict(rules_copy.get("page2") or {})
    examples = dict(rules_copy.get("examples") or {})
    hidden_example = dict(examples.get("hidden_single_step") or {})
    naked_example = dict(examples.get("naked_single_step") or {})

    page_width = frame.page_width
    page_height = frame.page_height
    content_left = frame.content_left
    content_width = frame.content_width
    center_x = content_left + (content_width / 2.0)

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    header_h = 92
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, page_width, header_h, stroke=0, fill=1)

    eyebrow_base = str((page1 if page_index == 1 else page2).get("eyebrow") or tr("sudoku_rules", language)).strip()
    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    _draw_blue_header_label(canvas, frame, _localized_series_header(eyebrow_base, page_index, page_total, language))

    if page_index == 1:
        headline = str(page1.get("headline") or "The rules are simple. The logic is elegant.").strip()
        kicker = str(page1.get("kicker") or "").strip()
        intro = str(page1.get("intro") or "").strip()
        rule_cards = list(page1.get("rule_cards") or [])
        mission_title = str(page1.get("mission_title") or "Your mission as a solver").strip()
        mission_body = str(page1.get("mission_body") or "").strip()
        principle_line = str(page1.get("principle_line") or "").strip()
        doors = list(page1.get("doors") or [])
        bridge = str(page1.get("bridge") or "").strip()

        title_block = draw_smart_headline(
            canvas,
            text=headline,
            font_name=fonts.bold,
            preferred_font_size=22.0,
            min_font_size=16.0,
            max_width=content_width * 0.92,
            x=center_x,
            first_baseline_y=page_height - 118,
            align="center",
            max_lines=2,
            leading_multiplier=1.10,
            fill_color=colors.black,
        )

        canvas.setFillColor(colors.HexColor("#4e4e4e"))
        canvas.setFont(fonts.italic, 10.8)
        kicker_y = title_block["bottom_y"] - 20
        kicker_lines = _wrap_text_to_width(
            kicker,
            font_name=fonts.italic,
            font_size=10.8,
            max_width=content_width * 0.92,
        )
        for idx, line in enumerate(kicker_lines):
            canvas.drawCentredString(center_x, kicker_y - idx * 13, line)

        kicker_bottom_y = kicker_y - ((len(kicker_lines) - 1) * 13) if kicker_lines else kicker_y
        intro_top = kicker_bottom_y - 26
        intro_h = 56
        _draw_soft_panel(canvas, x=content_left, y=intro_top - intro_h, w=content_width, h=intro_h, fill_hex="#f7f8fb")

        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 10.2)
        yy = intro_top - 18
        for line in _wrap_text_to_width(intro, font_name=fonts.regular, font_size=10.2, max_width=content_width - 32):
            canvas.drawString(content_left + 16, yy, line)
            yy -= 13

        cards_top = intro_top - intro_h - 16
        card_gap = 10
        card_w = (content_width - 2 * card_gap) / 3.0
        card_h = 86

        for idx, card in enumerate(rule_cards[:3]):
            cx = content_left + idx * (card_w + card_gap)
            _draw_soft_panel(canvas, x=cx, y=cards_top - card_h, w=card_w, h=card_h, fill_hex="#f6f7fb")
            canvas.setFillColor(colors.HexColor("#1f3c88"))
            canvas.setFont(fonts.bold, 11.5)
            canvas.drawString(cx + 12, cards_top - 18, str(card.get("title") or "").strip())
            canvas.setFillColor(colors.black)
            canvas.setFont(fonts.regular, 9.5)
            yb = cards_top - 36
            for line in _wrap_text_to_width(str(card.get("body") or "").strip(), font_name=fonts.regular, font_size=9.5, max_width=card_w - 22):
                canvas.drawString(cx + 12, yb, line)
                yb -= 11

        mission_top = cards_top - card_h - 18
        mission_h = 76
        _draw_soft_panel(canvas, x=content_left, y=mission_top - mission_h, w=content_width, h=mission_h, fill_hex="#eef3fb")
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 12.0)
        canvas.drawString(content_left + 16, mission_top - 18, mission_title)

        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 10.0)
        yy = mission_top - 36
        for line in _wrap_text_to_width(mission_body, font_name=fonts.regular, font_size=10.0, max_width=content_width - 32):
            canvas.drawString(content_left + 16, yy, line)
            yy -= 12

        principle_top = mission_top - mission_h - 14
        principle_h = 46
        _draw_soft_panel(canvas, x=content_left, y=principle_top - principle_h, w=content_width, h=principle_h, fill_hex="#fff4e8")
        canvas.setFillColor(colors.HexColor("#9a3412"))
        canvas.setFont(fonts.bold, 13.0)
        for idx, line in enumerate(
            _wrap_text_to_width(principle_line, font_name=fonts.bold, font_size=13.0, max_width=content_width - 28)
        ):
            canvas.drawCentredString(center_x, principle_top - 20 - idx * 14, line)

        doors_top = principle_top - principle_h - 16
        door_gap = 14
        door_w = (content_width - door_gap) / 2.0
        door_h = 120

        for idx, door in enumerate(doors[:2]):
            dx = content_left + idx * (door_w + door_gap)
            _draw_soft_panel(canvas, x=dx, y=doors_top - door_h, w=door_w, h=door_h, fill_hex="#f7f8fb")
            canvas.setFillColor(colors.HexColor("#1f3c88"))
            canvas.setFont(fonts.bold, 10.0)
            canvas.drawString(dx + 14, doors_top - 18, _friendly_rule_label(str(door.get("formal") or "").strip(), language))

            canvas.setFillColor(colors.black)
            canvas.setFont(fonts.bold, 13.0)
            yy = doors_top - 36
            for line in _wrap_text_to_width(str(door.get("title") or "").strip(), font_name=fonts.bold, font_size=13.0, max_width=door_w - 28):
                canvas.drawString(dx + 14, yy, line)
                yy -= 15

            yy -= 2
            canvas.setFont(fonts.regular, 9.7)
            for line in _wrap_text_to_width(str(door.get("body") or "").strip(), font_name=fonts.regular, font_size=9.7, max_width=door_w - 28):
                canvas.drawString(dx + 14, yy, line)
                yy -= 12

        if bridge:
            canvas.setFillColor(colors.HexColor("#4a4a4a"))
            canvas.setFont(fonts.italic, 9.7)
            by = _front_matter_bottom_note_y(frame)
            for line in _wrap_text_to_width(bridge, font_name=fonts.italic, font_size=9.7, max_width=content_width):
                canvas.drawString(content_left, by, line)
                by -= 11

        return

    headline = str(page2.get("headline") or "Two final truths, shown on a real puzzle").strip()
    kicker = str(page2.get("kicker") or "").strip()
    source_note = _format_frontmatter_puzzle_reference_line(
        str(page2.get("source_note") or examples.get("source") or "").strip(),
        language,
    )
    hidden_copy = dict(page2.get("hidden_single") or {})
    naked_copy = dict(page2.get("naked_single") or {})
    transition = str(page2.get("transition") or page2.get("footer") or "").strip()

    title_block = draw_smart_headline(
        canvas,
        text=headline,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=content_width * 0.92,
        x=center_x,
        first_baseline_y=page_height - 118,
        align="center",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    canvas.setFillColor(colors.HexColor("#4e4e4e"))
    canvas.setFont(fonts.italic, 10.8)
    kicker_y = title_block["bottom_y"] - 20

    kicker_lines = _wrap_text_to_width(
        kicker,
        font_name=fonts.italic,
        font_size=10.8,
        max_width=content_width * 0.92,
    )

    for idx, line in enumerate(kicker_lines):
        canvas.drawCentredString(center_x, kicker_y - idx * 13, line)

    if kicker_lines:
        kicker_bottom_y = kicker_y - ((len(kicker_lines) - 1) * 13)
    else:
        kicker_bottom_y = kicker_y

    source_y = kicker_bottom_y - 18
    if source_note:
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 9.7)
        source_lines = _wrap_text_to_width(
            source_note,
            font_name=fonts.bold,
            font_size=9.7,
            max_width=content_width * 0.92,
        )
        for idx, line in enumerate(source_lines):
            canvas.drawCentredString(center_x, source_y - idx * 12, line)

        if source_lines:
            source_bottom_y = source_y - ((len(source_lines) - 1) * 12)
        else:
            source_bottom_y = source_y

        panel1_top = source_bottom_y - 12
    else:
        panel1_top = source_y - 10

    panel_h = 224

    hidden_primary_house = dict(hidden_example.get("primary_house") or {})
    hidden_witness_houses = list(hidden_example.get("witness_houses") or [])

    hidden_rows = []
    hidden_cols = []
    hidden_boxes = []

    if str(hidden_primary_house.get("type") or "") == "row":
        hidden_rows.append(int(hidden_primary_house.get("index1to9") or 0))
    elif str(hidden_primary_house.get("type") or "") == "col":
        hidden_cols.append(int(hidden_primary_house.get("index1to9") or 0))
    elif str(hidden_primary_house.get("type") or "") == "box":
        hidden_boxes.append(int(hidden_primary_house.get("index1to9") or 0))

    for house in hidden_witness_houses:
        h_type = str((house or {}).get("type") or "")
        h_idx = int((house or {}).get("index1to9") or 0)
        if h_type == "row":
            hidden_rows.append(h_idx)
        elif h_type == "col":
            hidden_cols.append(h_idx)
        elif h_type == "box":
            hidden_boxes.append(h_idx)

    _draw_rules_example_panel(
        canvas,
        x=content_left,
        y_top=panel1_top,
        w=content_width,
        h=panel_h,
        fonts=fonts,
        title=str(hidden_copy.get("title") or "").strip(),
        formal=str(hidden_copy.get("formal") or "").strip(),
        body=str(hidden_copy.get("body") or "").strip(),
        facts=list(hidden_copy.get("facts") or []),
        board_kwargs={
            "grid81": str(hidden_example.get("grid81_before") or ""),
            "highlight_rows": hidden_rows,
            "highlight_cols": hidden_cols,
            "highlight_boxes": hidden_boxes,
            "target_cells": [hidden_example.get("target_cell")],
            "witness_cells": list(hidden_example.get("witness_cells") or []),
            "candidate_cells": list(hidden_example.get("candidate_cells_in_primary_house") or []),
            "result_cell": hidden_example.get("target_cell"),
            "result_digit": hidden_example.get("digit"),
            "language": language,
        },
        candidate_zoom=None,
    )

    panel2_top = panel1_top - panel_h - 16
    naked_acting_houses = list(naked_example.get("acting_houses") or [])

    naked_rows = []
    naked_cols = []
    naked_boxes = []

    for house in naked_acting_houses:
        h_type = str((house or {}).get("type") or "")
        h_idx = int((house or {}).get("index1to9") or 0)
        if h_type == "row":
            naked_rows.append(h_idx)
        elif h_type == "col":
            naked_cols.append(h_idx)
        elif h_type == "box":
            naked_boxes.append(h_idx)

    _draw_rules_example_panel(
        canvas,
        x=content_left,
        y_top=panel2_top,
        w=content_width,
        h=panel_h,
        fonts=fonts,
        title=str(naked_copy.get("title") or "").strip(),
        formal=str(naked_copy.get("formal") or "").strip(),
        body=str(naked_copy.get("body") or "").strip(),
        facts=list(naked_copy.get("facts") or []),
        board_kwargs={
            "grid81": str(naked_example.get("grid81_before") or ""),
            "highlight_rows": naked_rows,
            "highlight_cols": naked_cols,
            "highlight_boxes": naked_boxes,
            "target_cells": [naked_example.get("target_cell")],
            "witness_cells": list(naked_example.get("witness_cells") or []),
            "candidate_cells": [],
            "result_cell": naked_example.get("target_cell"),
            "result_digit": naked_example.get("digit"),
            "language": language,
        },
        candidate_zoom=None,
    )

    if transition:
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 9.6)
        by = _front_matter_bottom_note_y(frame)
        for line in _wrap_text_to_width(transition, font_name=fonts.italic, font_size=9.6, max_width=content_width):
            canvas.drawString(content_left, by, line)
            by -= 11


def render_tutorial_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    fonts = resolve_font_pack(payload.get("font_family"))
    tutorial_copy = _get_editorial_copy(payload, "tutorial")
    examples = dict(tutorial_copy.get("examples") or {})

    page_index = int(payload.get("tutorial_page_index") or payload.get("page_occurrence_index") or 1)
    page_total = int(payload.get("tutorial_page_total") or payload.get("page_occurrence_total") or 1)
    page_cfg = dict(tutorial_copy.get(f"page{page_index}") or tutorial_copy.get("page1") or {})
    language = _payload_language(payload)

    page_width = frame.page_width
    page_height = frame.page_height
    content_left = frame.content_left
    content_width = frame.content_width
    center_x = content_left + (content_width / 2.0)

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    header_h = 92
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, page_width, header_h, stroke=0, fill=1)

    eyebrow = str(page_cfg.get("eyebrow") or tr("tutorial", language)).strip()
    headline = str(page_cfg.get("headline") or tr("tutorial", language)).strip()
    kicker = str(page_cfg.get("kicker") or "").strip()
    intro = str(page_cfg.get("intro") or "").strip()
    source_note = _format_frontmatter_puzzle_reference_line(
        str(page_cfg.get("source_note") or "").strip(),
        language,
    )
    bridge = str(page_cfg.get("footer") or page_cfg.get("bridge") or "").strip()
    cards = list(page_cfg.get("cards") or page_cfg.get("examples") or [])

    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    _draw_blue_header_label(canvas, frame, _localized_series_header(eyebrow, page_index, page_total, language))

    title_block = draw_smart_headline(
        canvas,
        text=headline,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=content_width * 0.92,
        x=center_x,
        first_baseline_y=page_height - 118,
        align="center",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    canvas.setFillColor(colors.HexColor("#4e4e4e"))
    canvas.setFont(fonts.italic, 10.8)
    kicker_y = title_block["bottom_y"] - 20

    kicker_lines = _wrap_text_to_width(
        kicker,
        font_name=fonts.italic,
        font_size=10.8,
        max_width=content_width * 0.92,
    )

    for idx, line in enumerate(kicker_lines):
        canvas.drawCentredString(center_x, kicker_y - idx * 13, line)

    if kicker_lines:
        kicker_bottom_y = kicker_y - ((len(kicker_lines) - 1) * 13)
    else:
        kicker_bottom_y = kicker_y

    source_y = kicker_bottom_y - 18
    if source_note:
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 9.7)
        source_lines = _wrap_text_to_width(
            source_note,
            font_name=fonts.bold,
            font_size=9.7,
            max_width=content_width * 0.92,
        )
        for idx, line in enumerate(source_lines):
            canvas.drawCentredString(center_x, source_y - idx * 12, line)

        if source_lines:
            source_bottom_y = source_y - ((len(source_lines) - 1) * 12)
        else:
            source_bottom_y = source_y

        intro_top = source_bottom_y - 12
    else:
        intro_top = source_y - 10

    intro_h = 64
    _draw_soft_panel(canvas, x=content_left, y=intro_top - intro_h, w=content_width, h=intro_h, fill_hex="#f7f8fb")
    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 10.0)
    yy = intro_top - 18
    for line in _wrap_text_to_width(intro, font_name=fonts.regular, font_size=10.0, max_width=content_width - 32):
        canvas.drawString(content_left + 16, yy, line)
        yy -= 12

    page_examples = dict(examples.get(f"page{page_index}") or {})
    panel1_top = intro_top - intro_h - 16
    panel_h = 198

    def _tutorial_example_key(card: dict, slot_index: int) -> str:
        """
        Keep the visual grid aligned with the story card.

        Page 1 is the subsets tutorial:
          - Naked Triple story -> triplet_example
          - Naked Pair story   -> pair_example

        Page 2 is the intersections tutorial:
          - Claiming Pair story -> claiming_example
          - Pointing Pair story -> pointing_example

        This avoids the old hard-coded page-2 order where the first card
        always pulled pointing_example and the second card always pulled
        claiming_example.
        """
        label = f"{card.get('formal') or ''} {card.get('title') or ''}".lower()

        if page_index == 1:
            if "triple" in label or "triplet" in label:
                return "triplet_example"
            if "pair" in label:
                return "pair_example"
            return "triplet_example" if slot_index == 0 else "pair_example"

        if "claiming" in label:
            return "claiming_example"
        if "pointing" in label:
            return "pointing_example"
        return "claiming_example" if slot_index == 0 else "pointing_example"

    def _draw_tutorial_card(card: dict, *, slot_index: int, y_top: float) -> None:
        example_key = _tutorial_example_key(card, slot_index)
        ex = dict(page_examples.get(example_key) or {})

        _draw_rules_example_panel(
            canvas,
            x=content_left,
            y_top=y_top,
            w=content_width,
            h=panel_h,
            fonts=fonts,
            title=str(card.get("title") or "").strip(),
            formal="",
            body=str(card.get("body") or "").strip(),
            facts=[],
            board_kwargs={
                "grid81": str(ex.get("grid81_before") or ""),
                "highlight_rows": list(ex.get("highlight_rows") or []),
                "highlight_cols": list(ex.get("highlight_cols") or []),
                "highlight_boxes": list(ex.get("highlight_boxes") or []),
                "pattern_cells": list(ex.get("pattern_cells") or []),
                "target_cells": list(ex.get("target_cells") or []),
                "witness_cells": list(ex.get("witness_cells") or []),
                "candidate_cells": [],
                "cell_candidates": dict(ex.get("cell_candidates") or {}),
                "pattern_locked_digits": {
                    **dict(ex.get("pattern_locked_digits") or {}),
                    **dict(ex.get("prior_hero_witness_locked_digits") or {}),
                },
                "result_cell": ex.get("result_cell"),
                "result_digit": ex.get("result_digit"),
                "language": language,
            },
            candidate_zoom=None,
        )

    if len(cards) >= 1:
        _draw_tutorial_card(cards[0], slot_index=0, y_top=panel1_top)

    if len(cards) >= 2:
        panel2_top = panel1_top - panel_h - 14
        _draw_tutorial_card(cards[1], slot_index=1, y_top=panel2_top)

    if bridge:
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 9.6)
        by = _front_matter_bottom_note_y(frame)
        for line in _wrap_text_to_width(bridge, font_name=fonts.italic, font_size=9.6, max_width=content_width):
            canvas.drawString(content_left, by, line)
            by -= 11


def render_warmup_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    fonts = resolve_font_pack(payload.get("font_family"))
    warmup_copy = _get_editorial_copy(payload, "warmup")
    examples = dict(warmup_copy.get("examples") or {})

    page_index = int(payload.get("warmup_page_index") or payload.get("page_occurrence_index") or 1)
    page_total = int(payload.get("warmup_page_total") or payload.get("page_occurrence_total") or 1)
    page_cfg = dict(warmup_copy.get(f"page{page_index}") or warmup_copy.get("page1") or {})
    page_example = dict(examples.get(f"page{page_index}") or {})
    language = _payload_language(payload)

    page_width = frame.page_width
    page_height = frame.page_height
    content_left = frame.content_left
    content_width = frame.content_width
    center_x = content_left + (content_width / 2.0)

    eyebrow = str(page_cfg.get("eyebrow") or tr("warmup", language)).strip()
    headline = str(page_cfg.get("headline") or tr("warmup", language)).strip()
    kicker = str(page_cfg.get("kicker") or "").strip()
    intro = str(page_cfg.get("intro") or "").strip()
    source_note = _format_frontmatter_puzzle_reference_line(
        str(page_cfg.get("source_note") or "").strip(),
        language,
    )
    prompt_cards = list(page_cfg.get("prompt_cards") or [])
    check_title = str(page_cfg.get("check_title") or "Check yourself").strip()
    check_lines = list(page_cfg.get("check_lines") or [])
    footer_note = str(page_cfg.get("footer_note") or "").strip()

    challenge1 = dict(page_example.get("challenge1") or {})
    challenge2 = dict(page_example.get("challenge2") or {})

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    header_h = 92
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, page_width, header_h, stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    canvas.drawString(content_left, _blue_header_label_y(frame), _localized_series_header(eyebrow, page_index, page_total, language))

    title_block = draw_smart_headline(
        canvas,
        text=headline,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=content_width * 0.92,
        x=center_x,
        first_baseline_y=page_height - 118,
        align="center",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    canvas.setFillColor(colors.HexColor("#4e4e4e"))
    canvas.setFont(fonts.italic, 10.8)
    kicker_y = title_block["bottom_y"] - 20
    kicker_lines = _wrap_text_to_width(
        kicker,
        font_name=fonts.italic,
        font_size=10.8,
        max_width=content_width * 0.92,
    )
    for idx, line in enumerate(kicker_lines):
        canvas.drawCentredString(center_x, kicker_y - idx * 13, line)

    kicker_bottom_y = kicker_y - ((len(kicker_lines) - 1) * 13) if kicker_lines else kicker_y
    source_y = kicker_bottom_y - 18
    if source_note:
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 9.7)
        for idx, line in enumerate(
            _wrap_text_to_width(source_note, font_name=fonts.bold, font_size=9.7, max_width=content_width * 0.92)
        ):
            canvas.drawCentredString(center_x, source_y - idx * 12, line)
        intro_top = source_y - 24
    else:
        intro_top = kicker_bottom_y - 28

    intro_h = 56
    _draw_soft_panel(canvas, x=content_left, y=intro_top - intro_h, w=content_width, h=intro_h, fill_hex="#f7f8fb")
    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 10.0)
    yy = intro_top - 18
    for line in _wrap_text_to_width(intro, font_name=fonts.regular, font_size=10.0, max_width=content_width - 32):
        canvas.drawString(content_left + 16, yy, line)
        yy -= 12

    def _draw_warmup_challenge_panel(*, y_top: float, challenge: Dict[str, Any], card: Dict[str, Any]) -> None:
        panel_h = 150
        _draw_soft_panel(
            canvas,
            x=content_left,
            y=y_top - panel_h,
            w=content_width,
            h=panel_h,
            fill_hex="#f7f8fb",
        )

        board_size = 126
        board_x = content_left + 14
        board_y_top = y_top - 12

        _draw_sudoku_board(
            canvas,
            x=board_x,
            y_top=board_y_top,
            size=board_size,
            grid81=str(challenge.get("grid81_before") or ""),
            fonts=fonts,
            highlight_rows=[],
            highlight_cols=[],
            highlight_boxes=[],
            target_cells=[],
            witness_cells=[],
            candidate_cells=[],
            pattern_cells=[],
            cell_candidates={},
            result_cell=None,
            result_digit=None,
        )

        text_x = board_x + board_size + 16
        text_w = content_width - (text_x - content_left) - 14
        yy = y_top - 18

        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.bold, 12.2)
        for line in _wrap_text_to_width(
            str(card.get("title") or "").strip(),
            font_name=fonts.bold,
            font_size=12.2,
            max_width=text_w,
        ):
            canvas.drawString(text_x, yy, line)
            yy -= 14

        yy -= 2
        canvas.setFont(fonts.regular, 9.6)
        for line in _wrap_text_to_width(
            str(card.get("body") or "").strip(),
            font_name=fonts.regular,
            font_size=9.6,
            max_width=text_w,
        ):
            canvas.drawString(text_x, yy, line)
            yy -= 11

    body_top = intro_top - intro_h - 14
    if len(prompt_cards) >= 1:
        _draw_warmup_challenge_panel(
            y_top=body_top,
            challenge=challenge1,
            card=prompt_cards[0],
        )

    if len(prompt_cards) >= 2:
        _draw_warmup_challenge_panel(
            y_top=body_top - 162,
            challenge=challenge2,
            card=prompt_cards[1],
        )

    check_y = 84
    check_h = 86
    _draw_soft_panel(canvas, x=content_left, y=check_y, w=content_width, h=check_h, fill_hex="#eef3fb")

    cx = content_left + 14
    cy = check_y + check_h - 18
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 10.2)
    canvas.drawString(cx, cy, check_title)
    cy -= 15

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 9.4)
    for idx, line_text in enumerate(check_lines[:3]):
        wrapped = _wrap_text_to_width(line_text, font_name=fonts.regular, font_size=9.4, max_width=content_width - 28)
        for line in wrapped:
            canvas.drawString(cx, cy, line)
            cy -= 10
        cy -= 2
        if idx < len(check_lines[:3]) - 1:
            cy -= 6

    if footer_note:
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 9.5)
        fy = _front_matter_bottom_note_y(frame)
        for line in _wrap_text_to_width(footer_note, font_name=fonts.italic, font_size=9.5, max_width=content_width):
            canvas.drawString(content_left, fy, line)
            fy -= 11



def _normalize_locale_for_last_words(payload: dict) -> str:
    metadata = dict(payload.get("publication_metadata") or {})
    layout_config = dict(payload.get("layout_config") or {})

    raw = str(
        metadata.get("locale")
        or payload.get("locale")
        or layout_config.get("language")
        or metadata.get("language")
        or "en"
    ).strip().lower()

    language_to_locale = {
        "english": "en",
        "german": "de",
        "deutsch": "de",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "italian": "it",
        "italiano": "it",
        "spanish": "es",
        "español": "es",
        "espanol": "es",
    }

    raw = language_to_locale.get(raw, raw)
    return raw.split("-")[0].split("_")[0] or "en"


def _resolve_solution_booklet_copy(
    *,
    payload: dict,
    ending_copy: dict,
) -> dict:
    ecosystem_config = dict(payload.get("ecosystem_config") or {})
    publication_metadata = dict(payload.get("publication_metadata") or {})

    locale = _normalize_locale_for_last_words(payload)

    book_code = str(
        ending_copy.get("book_code")
        or payload.get("book_code")
        or publication_metadata.get("book_code")
        or ""
    ).strip()

    support_email = str(
        ending_copy.get("solution_booklet_email")
        or ecosystem_config.get("support_email")
        or "gfotso@umich.edu"
    ).strip()

    localized = {
        "en": {
            "title": "Request a solution booklet for this book",
            "body": (
                "Your copy of this book may not include printed solutions at the end. "
                "If it does not, you may request a printable solution booklet by emailing "
                "{email}. The book support and online access tools already provide "
                "step-by-step solutions; this booklet is an additional final-answer reference. "
                "Please include the book code in your email subject: {book_code}."
            ),
            "book_code_label": "Book code",
        },
        "de": {
            "title": "Lösungsheft für dieses Buch anfordern",
            "body": (
                "Ihr Exemplar dieses Buches enthält möglicherweise keine gedruckten Lösungen am Ende. "
                "Falls nicht, können Sie ein druckbares Lösungsheft per E-Mail an {email} anfordern. "
                "Die Buchhilfe und die Online-Zugangswerkzeuge bieten bereits Schritt-für-Schritt-Lösungen; "
                "dieses Heft ist eine zusätzliche Übersicht mit den Endlösungen. "
                "Bitte geben Sie den Buchcode im Betreff Ihrer E-Mail an: {book_code}."
            ),
            "book_code_label": "Buchcode",
        },
        "fr": {
            "title": "Demander le livret de solutions de ce livre",
            "body": (
                "Votre exemplaire de ce livre peut ne pas inclure les solutions imprimées à la fin. "
                "Si c’est le cas, vous pouvez demander un livret de solutions imprimable en envoyant un e-mail à {email}. "
                "L’assistance du livre et les outils d’accès en ligne proposent déjà des solutions étape par étape ; "
                "ce livret est une référence supplémentaire avec les solutions finales. "
                "Veuillez indiquer le code du livre dans l’objet de votre e-mail : {book_code}."
            ),
            "book_code_label": "Code du livre",
        },
        "es": {
            "title": "Solicite el cuaderno de soluciones de este libro",
            "body": (
                "Es posible que su ejemplar de este libro no incluya las soluciones impresas al final. "
                "Si no las incluye, puede solicitar un cuaderno de soluciones imprimible escribiendo a {email}. "
                "La ayuda del libro y las herramientas de acceso en línea ya ofrecen soluciones paso a paso; "
                "este cuaderno es una referencia adicional con las soluciones finales. "
                "Indique el código del libro en el asunto de su correo electrónico: {book_code}."
            ),
            "book_code_label": "Código del libro",
        },
        "it": {
            "title": "Richiedere il libretto delle soluzioni di questo libro",
            "body": (
                "La vostra copia di questo libro potrebbe non includere le soluzioni stampate alla fine. "
                "In tal caso, potete richiedere un libretto delle soluzioni stampabile scrivendo a {email}. "
                "Il supporto del libro e gli strumenti di accesso online offrono già soluzioni passo passo; "
                "questo libretto è un riferimento aggiuntivo con le soluzioni finali. "
                "Indicate il codice del libro nell’oggetto della vostra e-mail: {book_code}."
            ),
            "book_code_label": "Codice del libro",
        },
    }

    defaults = localized.get(locale, localized["en"])

    title = str(
        ending_copy.get("solution_booklet_title")
        or defaults["title"]
    ).strip()

    body_template = str(
        ending_copy.get("solution_booklet_body")
        or defaults["body"]
    ).strip()

    safe_book_code = book_code or "BXX-en"

    body = (
        body_template
        .replace("{email}", support_email)
        .replace("{book_code}", safe_book_code)
    )

    return {
        "enabled": bool(title or body),
        "title": title,
        "body": body,
        "email": support_email,
        "book_code": book_code,
        "book_code_label": str(
            ending_copy.get("book_code_label")
            or defaults["book_code_label"]
        ).strip(),
    }

def render_promo_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    page_width = frame.page_width
    page_height = frame.page_height
    fonts = resolve_font_pack(payload.get("font_family"))
    ending_copy = _get_editorial_copy(payload, "ending")

    eyebrow = str(ending_copy.get("eyebrow") or "Last Words").strip()
    headline = str(ending_copy.get("headline") or payload.get("title") or "Thank you").strip()

    body_paragraphs = list(ending_copy.get("body_paragraphs") or [])
    if not body_paragraphs:
        fallback = str(payload.get("body") or "").strip()
        if fallback:
            body_paragraphs = [fallback]

    review_title = str(ending_copy.get("review_title") or "Review this book").strip()
    review_body = str(ending_copy.get("review_body") or "If you would like to leave a review, please visit:").strip()
    review_url = _build_review_url(
        payload,
        fallback_url=str(ending_copy.get("review_url") or "contextionary.com/review100").strip(),
    )


    solution_booklet_copy = _resolve_solution_booklet_copy(
        payload=payload,
        ending_copy=ending_copy,
    )

    book_code = solution_booklet_copy["book_code"]
    solution_booklet_title = solution_booklet_copy["title"]
    solution_booklet_body = solution_booklet_copy["body"]
    solution_booklet_book_code_label = solution_booklet_copy["book_code_label"]

    support_title = str(ending_copy.get("support_title") or "").strip()
    support_body = str(ending_copy.get("support_body") or "").strip()
    support_url_label = str(ending_copy.get("support_url_label") or "Website access:").strip()
    support_url = _build_solution_support_url(
        payload,
        fallback_url=str(ending_copy.get("support_url") or "").strip(),
    )
    support_same_note = str(
        ending_copy.get("support_same_note")
        or "The website link and the QR code both open the same Solution App destination for this book."
    ).strip()

    support_qr_label = str(ending_copy.get("support_qr_label") or "QR access").strip()
    support_qr_caption = str(
        ending_copy.get("support_qr_caption") or "Scan to open the same Solution App for this book."
    ).strip()
    support_qr_image_path = str(ending_copy.get("support_qr_image_path") or "").strip()
    support_qr_path = _resolve_asset_path(support_qr_image_path) if support_qr_image_path else None

    closing_signature = str(ending_copy.get("closing_signature") or "").strip()

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    header_h = 92
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, page_width, header_h, stroke=0, fill=1)

    content_left = frame.content_left
    content_width = frame.content_width
    center_x = content_left + (content_width / 2.0)

    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    _draw_blue_header_label(canvas, frame, eyebrow)

    title_block = draw_smart_headline(
        canvas,
        text=headline,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=content_width * 0.92,
        x=center_x,
        first_baseline_y=page_height - 118,
        align="center",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.black,
    )

    story_top = title_block["bottom_y"] - 22
    story_h = 148

    review_gap = 12
    review_top = story_top - story_h - review_gap
    review_h = 72

    solution_booklet_enabled = bool(solution_booklet_title or solution_booklet_body)
    solution_booklet_gap = 12
    solution_booklet_h = 112 if solution_booklet_enabled else 0
    solution_booklet_top = (
        review_top - review_h - solution_booklet_gap
        if solution_booklet_enabled
        else None
    )

    support_enabled = bool(support_title or support_body or support_url or support_qr_path)
    support_gap = 12
    support_h = 124 if support_enabled and support_qr_path else 88
    support_top = (
        solution_booklet_top - solution_booklet_h - support_gap
        if support_enabled and solution_booklet_top is not None
        else review_top - review_h - support_gap
        if support_enabled
        else None
    )

    _draw_soft_panel(
        canvas,
        x=content_left,
        y=story_top - story_h,
        w=content_width,
        h=story_h,
        fill_hex="#f7f8fb",
    )

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(content_left, story_top - story_h, 5, story_h, stroke=0, fill=1)

    _draw_paragraph_block(
        canvas,
        x=content_left + 20,
        y_top=story_top - 20,
        width=content_width - 40,
        paragraphs=body_paragraphs,
        font_name=fonts.regular,
        font_size=10.25,
        leading=13.2,
        paragraph_gap=7.5,
        color=colors.black,
    )

    _draw_soft_panel(
        canvas,
        x=content_left,
        y=review_top - review_h,
        w=content_width,
        h=review_h,
        fill_hex="#eef3fb",
    )

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(content_left, review_top - review_h, 5, review_h, stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 12.2)
    canvas.drawString(content_left + 18, review_top - 18, review_title)

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 10.0)
    review_body_y = review_top - 38
    for line in _wrap_text_to_width(
        review_body,
        font_name=fonts.regular,
        font_size=10.0,
        max_width=content_width - 36,
    ):
        canvas.drawString(content_left + 18, review_body_y, line)
        review_body_y -= 12

    _draw_label_and_link(
        canvas,
        x=content_left + 18,
        y=review_top - 68,
        label="",
        link_text=review_url,
        fonts=fonts,
        label_font_size=10.2,
        link_font_size=10.2,
    )


    if solution_booklet_enabled and solution_booklet_top is not None:
        _draw_soft_panel(
            canvas,
            x=content_left,
            y=solution_booklet_top - solution_booklet_h,
            w=content_width,
            h=solution_booklet_h,
            fill_hex="#fff8ec",
        )

        canvas.setFillColor(colors.HexColor("#d86b21"))
        canvas.rect(content_left, solution_booklet_top - solution_booklet_h, 5, solution_booklet_h, stroke=0, fill=1)

        inner_x = content_left + 18
        inner_y_top = solution_booklet_top - 18
        inner_w = content_width - 36

        canvas.setFillColor(colors.HexColor("#8a3f0b"))
        canvas.setFont(fonts.bold, 12.0)
        canvas.drawString(inner_x, inner_y_top, solution_booklet_title)

        body_y = inner_y_top - 19
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.3)
        for line in _wrap_text_to_width(
            solution_booklet_body,
            font_name=fonts.regular,
            font_size=9.3,
            max_width=inner_w,
        )[:7]:
            canvas.drawString(inner_x, body_y, line)
            body_y -= 10.5

        if book_code:
            canvas.setFillColor(colors.HexColor("#8a3f0b"))
            canvas.setFont(fonts.bold, 9.4)
            canvas.drawString(
                inner_x,
                solution_booklet_top - solution_booklet_h + 14,
                f"{solution_booklet_book_code_label}: {book_code}",
            )





    if support_enabled and support_top is not None:
        _draw_soft_panel(
            canvas,
            x=content_left,
            y=support_top - support_h,
            w=content_width,
            h=support_h,
            fill_hex="#f7f8fb",
        )

        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.rect(content_left, support_top - support_h, 5, support_h, stroke=0, fill=1)

        inner_x = content_left + 18
        inner_y_top = support_top - 18
        inner_w = content_width - 36

        qr_card_w = 126 if support_qr_path else 0
        split_gap = 16 if support_qr_path else 0
        text_w = inner_w - qr_card_w - split_gap if support_qr_path else inner_w

        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 12.2)
        canvas.drawString(inner_x, inner_y_top, support_title or "Book support and online access")

        body_y = inner_y_top - 20
        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.7)
        for line in _wrap_text_to_width(
            support_body,
            font_name=fonts.regular,
            font_size=9.7,
            max_width=text_w,
        ):
            canvas.drawString(inner_x, body_y, line)
            body_y -= 11

        if support_url:
            body_y -= 4
            _draw_label_and_link(
                canvas,
                x=inner_x,
                y=body_y,
                label=support_url_label,
                link_text=support_url,
                fonts=fonts,
                label_font_size=9.7,
                link_font_size=9.7,
            )
            body_y -= 16

        note_h = 24
        note_w = text_w
        _draw_soft_panel(
            canvas,
            x=inner_x,
            y=body_y - note_h + 7,
            w=note_w,
            h=note_h,
            fill_hex="#eef3fb",
        )
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 8.7)
        note_lines = _wrap_text_to_width(
            support_same_note,
            font_name=fonts.italic,
            font_size=8.7,
            max_width=note_w - 12,
        )
        note_text_y = body_y - 3
        for idx, line in enumerate(note_lines[:2]):
            canvas.drawString(inner_x + 6, note_text_y - (idx * 9), line)

        if support_qr_path:
            qr_card_x = inner_x + text_w + split_gap
            qr_card_y = support_top - support_h + 14
            qr_card_h = support_h - 28

            canvas.setFillColor(colors.white)
            canvas.setStrokeColor(colors.HexColor("#d7deeb"))
            canvas.setLineWidth(0.9)
            canvas.rect(qr_card_x, qr_card_y, qr_card_w, qr_card_h, stroke=1, fill=1)

            canvas.setFillColor(colors.HexColor("#1f3c88"))
            canvas.setFont(fonts.bold, 9.3)
            canvas.drawCentredString(qr_card_x + (qr_card_w / 2.0), qr_card_y + qr_card_h - 12, support_qr_label)

            qr_size = 72
            qr_x = qr_card_x + (qr_card_w - qr_size) / 2.0
            qr_y = qr_card_y + 26

            try:
                canvas.drawImage(
                    ImageReader(str(support_qr_path)),
                    qr_x,
                    qr_y,
                    width=qr_size,
                    height=qr_size,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                canvas.setFillColor(colors.HexColor("#999999"))
                canvas.setFont(fonts.italic, 8.8)
                canvas.drawCentredString(
                    qr_card_x + (qr_card_w / 2.0),
                    qr_y + (qr_size / 2.0),
                    "QR unavailable",
                )

            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.setFont(fonts.regular, 8.0)
            caption_y = qr_card_y + 16
            caption_lines = _wrap_text_to_width(
                support_qr_caption,
                font_name=fonts.regular,
                font_size=8.0,
                max_width=qr_card_w - 12,
            )
            for idx, line in enumerate(caption_lines[:2]):
                canvas.drawCentredString(qr_card_x + (qr_card_w / 2.0), caption_y - (idx * 8.5), line)

    if closing_signature:
        rule_y = 80
        canvas.setStrokeColor(colors.HexColor("#dddddd"))
        canvas.setLineWidth(0.8)
        canvas.line(content_left, rule_y, content_left + content_width, rule_y)

        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 10.0)
        sig_y = 68
        for idx, line in enumerate(closing_signature.split("\n")):
            canvas.drawString(content_left, sig_y - (idx * 12), line)