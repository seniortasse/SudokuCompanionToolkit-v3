from __future__ import annotations

from typing import Any, Dict, List, Sequence

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas

from python.publishing.i18n.strings import tr
from python.publishing.techniques.technique_catalog import (
    get_public_technique_name,
    public_combo_label,
)
from .headline_layout import draw_smart_headline
from .page_geometry import PageFrame
from .typography import resolve_font_pack


# KDP-safe blue-header label baseline.
# Position from trim_top, not page_height, because bleed pages have a larger MediaBox.
_BLUE_HEADER_LABEL_FROM_TRIM_TOP_PT = 54.0


def _blue_header_label_y(frame: PageFrame) -> float:
    return frame.trim_top - _BLUE_HEADER_LABEL_FROM_TRIM_TOP_PT


def _draw_blue_header_label(canvas: Canvas, frame: PageFrame, text: str) -> None:
    if frame.mirror_margins and not frame.is_even_page:
        canvas.drawRightString(frame.content_right, _blue_header_label_y(frame), str(text))
    else:
        canvas.drawString(frame.content_left, _blue_header_label_y(frame), str(text))


def _wrap_text(value: str, width: int) -> List[str]:
    words = str(value or "").split()
    if not words:
        return []

    lines: List[str] = []
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



def _section_key_from_payload(payload: Dict[str, Any]) -> str:
    """
    Resolve the editorial_copy.sections key for the current section.

    Important:
    section_code values such as L1/L2/L3 are order labels, not difficulty
    identities. Different books can map them differently:

        B02: L1=medium, L2=hard,   L3=expert
        B03: L1=easy,   L2=medium, L3=hard

    Therefore we must prefer explicit semantic labels first and only use
    section_code as a last-resort fallback.
    """
    explicit = str(
        payload.get("section_key")
        or payload.get("difficulty_key")
        or payload.get("difficulty")
        or payload.get("difficulty_label")
        or payload.get("difficulty_label_hint")
        or payload.get("title")
        or ""
    ).strip().lower()

    normalized = (
        explicit
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    known = {
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
        "expert": "expert",
        "medium to hard": "medium",
        "medium to expert": "medium",
        "easy to hard": "easy",
        "easy to expert": "easy",
    }

    if normalized in known:
        return known[normalized]

    # Handle labels like "L1 Easy", "Easy Section", "section easy", etc.
    tokens = {part.strip() for part in normalized.split() if part.strip()}
    for key in ("easy", "medium", "hard", "expert"):
        if key in tokens:
            return key

    # Last-resort fallback only. Do not map L1/L2/L3 to difficulty here.
    code = str(payload.get("section_code") or "").strip().lower()
    return code or normalized or "section"


def _section_copy_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    editorial = dict(payload.get("editorial_copy") or {})
    sections = dict(editorial.get("sections") or {})

    section_key = _section_key_from_payload(payload)
    copy = dict(sections.get(section_key) or {})
    if copy:
        return copy

    # Defensive fallback: if the resolved key failed, try the visible title.
    # This catches cases where payload carries title="Easy" but no section_key.
    title_key = str(payload.get("title") or "").strip().lower()
    if title_key and title_key in sections:
        return dict(sections.get(title_key) or {})

    hint_key = str(payload.get("difficulty_label_hint") or "").strip().lower()
    if hint_key and hint_key in sections:
        return dict(sections.get(hint_key) or {})

    return {}


def _humanize_technique(value: str, language: str = "en") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    if "+" in raw:
        return public_combo_label(raw, plural=True)

    # A few existing localized labels are already available through i18n.
    # Keep those where they exist; fall back to the canonical commercial
    # catalog for the rest.
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

    value = str(localized.get(key) or "").strip()
    if value:
        return value

    return get_public_technique_name(raw, plural=True)


def _unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def _format_top_combo_lines(items: Sequence[str], language: str = "en") -> list[str]:
    formatted: list[str] = []
    for raw in items:
        value = str(raw or "").strip()
        if not value:
            continue

        if "+" in value:
            human = public_combo_label(value, plural=True)
            parts = [part.strip() for part in human.split("+") if part.strip()]
            unique_parts = _unique_preserve_order(parts)

            if len(unique_parts) == 1:
                formatted.append(f"Back-to-back {unique_parts[0]}")
            else:
                formatted.append(" + ".join(unique_parts))
        else:
            formatted.append(_humanize_technique(value, language))

    return _unique_preserve_order(formatted)


def _draw_mask_preview(canvas: Canvas, *, x: float, y: float, size: float, mask81: str) -> None:
    if not mask81 or len(mask81) != 81:
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.rect(x, y, size, size, stroke=1, fill=0)
        return

    cell = size / 9.0

    for r in range(9):
        for c in range(9):
            if mask81[r * 9 + c] == "1":
                x0 = x + c * cell
                y0 = y + size - (r + 1) * cell
                canvas.setFillColor(colors.black)
                canvas.rect(x0, y0, cell, cell, stroke=0, fill=1)

    for i in range(10):
        width = 1.4 if i % 3 == 0 else 0.4
        canvas.setLineWidth(width)
        canvas.setStrokeColor(colors.HexColor("#777777"))
        xx = x + i * cell
        yy = y + i * cell
        canvas.line(xx, y, xx, y + size)
        canvas.line(x, yy, x + size, yy)


def render_section_highlights_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    fonts = resolve_font_pack(payload.get("font_family"))
    page_height = frame.page_height
    section_copy = _section_copy_from_payload(payload)
    language = str(payload.get("language") or "en")

    title = str(
        payload.get("headline")
        or section_copy.get("highlights_headline")
        or f"{payload.get('title', 'Section')} Highlights"
    ).strip()
    kicker = str(section_copy.get("highlights_kicker") or "").strip()
    story = str(section_copy.get("story") or payload.get("story") or "").strip()

    definition_title = str(section_copy.get("definition_title") or "What this section means").strip()
    definition_body = str(section_copy.get("definition_body") or "").strip()
    expectation_title = str(section_copy.get("expectation_title") or "What to expect").strip()
    expectation_body = str(section_copy.get("expectation_body") or "").strip()
    stats_intro = str(section_copy.get("stats_intro") or "").strip()
    technique_list_title = str(section_copy.get("technique_list_title") or "What you will encounter").strip()
    technique_list_intro = str(section_copy.get("technique_list_intro") or "").strip()
    closing_line = str(section_copy.get("closing_line") or "").strip()

    technique_list = _unique_preserve_order(
        [_humanize_technique(x, language) for x in list(section_copy.get("technique_list_engine_ids") or [])]
    )
    if not technique_list:
        technique_list = list(section_copy.get("technique_list") or [])

    top_techniques = _unique_preserve_order(
        [_humanize_technique(x, language) for x in list(payload.get("top_techniques") or [])]
    )
    top_combos = _format_top_combo_lines(list(payload.get("top_combos") or []), language)

    if not technique_list:
        technique_list = top_techniques[:]

    stats_cards = [
        (tr("puzzles_in_section", language), str(payload.get("puzzle_count") or "—")),
        (tr("weight_range", language), _range_label(payload.get("weight_min"), payload.get("weight_max"))),
        (tr("clue_range", language), _range_label(payload.get("clue_min"), payload.get("clue_max"))),
        (
            tr("pattern_variety", language),
            f"{payload.get('unique_patterns') or '—'} {tr('patterns_word', language)} • {payload.get('unique_pattern_families') or '—'} {tr('families_word', language)}",
        ),
    ]

    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, frame.page_width, frame.page_height, stroke=0, fill=1)

    header_h = 92
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(0, page_height - header_h, frame.page_width, header_h, stroke=0, fill=1)

    eyebrow = f"{payload.get('title', 'Section')} • {tr('section_highlights', language)}"
    canvas.setFillColor(colors.HexColor("#dfe7fb"))
    canvas.setFont(fonts.bold, 10.2)
    _draw_blue_header_label(canvas, frame, eyebrow)

    content_width = frame.content_width
    center_x = frame.content_left + (content_width / 2.0)

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

    kicker_lines = _wrap_text(kicker, 94)[:2] if kicker else []
    if kicker_lines:
        canvas.setFillColor(colors.HexColor("#4e4e4e"))
        canvas.setFont(fonts.italic, 10.8)
        kicker_y = title_block["bottom_y"] - 20
        for idx, line in enumerate(kicker_lines):
            canvas.drawCentredString(center_x, kicker_y - (idx * 13), line)
        body_top = kicker_y - ((len(kicker_lines) - 1) * 13) - 22
    else:
        body_top = title_block["bottom_y"] - 28

    # ------------------------------------------------------------------
    # Footer zone (reserved first, so body never collides with it)
    #
    # Keep the closing line safely above the printed page footer.
    # The bottom flavor panel is intentionally compact; it is a quick
    # section summary, not a large content block.
    # ------------------------------------------------------------------
    footer_bottom_y = 46
    footer_line_gap = 10

    closing_lines = _wrap_text(closing_line, 108)[:2] if closing_line else []
    closing_h = (len(closing_lines) * footer_line_gap) if closing_lines else 0

    tech_line = ", ".join(top_techniques[:4]) if top_techniques else ""
    tech_lines = _wrap_text(tech_line, 46)[:2] if tech_line else []

    combo_line = ", ".join(top_combos[:3]) if top_combos else ""
    combo_lines = _wrap_text(combo_line, 46)[:2] if combo_line else []

    # Compact panel: title row + up to two short body lines.
    flavor_h = 42

    footer_top = footer_bottom_y + closing_h + (8 if closing_lines else 0)
    flavor_y = footer_top
    flavor_top = flavor_y + flavor_h

    story_lines = _wrap_text(story, 102)
    story_h = 16 + (len(story_lines) * 12) + 14

    story_top = body_top
    _draw_soft_panel(
        canvas,
        x=frame.content_left,
        y=story_top - story_h,
        w=frame.content_width,
        h=story_h,
        fill_hex="#f7f8fb",
    )

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 10.0)
    yy = story_top - 16
    for line in story_lines:
        canvas.drawString(frame.content_left + 16, yy, line)
        yy -= 12

    cards_top = story_top - story_h - 14
    gap = 12
    card_w = (frame.content_width - gap) / 2.0

    definition_lines = _wrap_text(definition_body, 41)
    expectation_lines = _wrap_text(expectation_body, 41)
    max_card_lines = max(len(definition_lines), len(expectation_lines), 1)
    card_h = 18 + 14 + 8 + (max_card_lines * 10) + 12

    for idx, (card_title, body_lines) in enumerate(
        [
            (definition_title, definition_lines),
            (expectation_title, expectation_lines),
        ]
    ):
        x = frame.content_left + idx * (card_w + gap)
        _draw_soft_panel(canvas, x=x, y=cards_top - card_h, w=card_w, h=card_h, fill_hex="#f7f8fb")

        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont(fonts.bold, 11.0)
        canvas.drawString(x + 14, cards_top - 18, card_title)

        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.regular, 9.2)
        yy = cards_top - 36
        for line in body_lines:
            canvas.drawString(x + 14, yy, line)
            yy -= 10

    stats_top = cards_top - card_h - 14
    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 12.0)
    canvas.drawString(frame.content_left, stats_top, tr("section_snapshot", language))

    stats_intro_lines = _wrap_text(stats_intro, 108)[:2] if stats_intro else []
    if stats_intro_lines:
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.setFont(fonts.regular, 9.0)
        for idx, line in enumerate(stats_intro_lines):
            canvas.drawString(frame.content_left, stats_top - 14 - (idx * 10), line)

    stats_y = stats_top - 30
    if stats_intro_lines:
        stats_y -= (len(stats_intro_lines) - 1) * 10

    stat_gap = 10
    stat_w = (frame.content_width - stat_gap) / 2.0
    stat_h = 38

    for idx, (label, value) in enumerate(stats_cards):
        x = frame.content_left + (idx % 2) * (stat_w + stat_gap)
        y_top = stats_y - (idx // 2) * (stat_h + 10)

        _draw_soft_panel(canvas, x=x, y=y_top - stat_h, w=stat_w, h=stat_h, fill_hex="#f8fafc")
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.setFont(fonts.regular, 9.0)
        canvas.drawString(x + 10, y_top - 12, label)

        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.bold, 10.8)
        canvas.drawString(x + 10, y_top - 26, str(value))

    stats_bottom = stats_y - (2 * stat_h) - 10

    technique_top = stats_bottom - 14
    technique_bottom = flavor_top + 12
    technique_panel_h = technique_top - technique_bottom

    _draw_soft_panel(
        canvas,
        x=frame.content_left,
        y=technique_bottom,
        w=frame.content_width,
        h=technique_panel_h,
        fill_hex="#eef3fb",
    )

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 11.6)
    canvas.drawString(frame.content_left + 14, technique_top - 18, technique_list_title)

    intro_lines = _wrap_text(technique_list_intro, 104)[:2] if technique_list_intro else []
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.setFont(fonts.regular, 9.0)
    intro_y = technique_top - 34
    for idx, line in enumerate(intro_lines):
        canvas.drawString(frame.content_left + 14, intro_y - (idx * 10), line)

    left_items = technique_list[::2]
    right_items = technique_list[1::2]

    list_start_y = technique_top - 34 - (len(intro_lines) * 10) - 10
    left_x = frame.content_left + 14
    right_x = frame.content_left + (frame.content_width / 2.0) + 8

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 9.2)

    yy = list_start_y
    for item in left_items:
        canvas.drawString(left_x, yy, u"\u2022")
        canvas.drawString(left_x + 14, yy, str(item))
        yy -= 10

    yy = list_start_y
    for item in right_items:
        canvas.drawString(right_x, yy, u"\u2022")
        canvas.drawString(right_x + 14, yy, str(item))
        yy -= 10

    _draw_soft_panel(
        canvas,
        x=frame.content_left,
        y=flavor_y,
        w=frame.content_width,
        h=flavor_h,
        fill_hex="#f8fafc",
    )

    left_block_x = frame.content_left + 14
    right_block_x = frame.content_left + (frame.content_width / 2.0) + 8

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, 10.2)
    left_flavor_title = str(
        section_copy.get("flavor_left_title") or tr("flavor_left_title", language)
    ).strip()
    right_flavor_title = str(
        section_copy.get("flavor_right_title") or tr("flavor_right_title", language)
    ).strip()

    canvas.drawString(left_block_x, flavor_top - 16, left_flavor_title)
    canvas.drawString(right_block_x, flavor_top - 16, right_flavor_title)

    canvas.setFillColor(colors.black)
    canvas.setFont(fonts.regular, 9.0)
    body_y = flavor_top - 31

    for idx, line in enumerate(tech_lines):
        canvas.drawString(left_block_x, body_y - (idx * 10), line)

    for idx, line in enumerate(combo_lines):
        canvas.drawString(right_block_x, body_y - (idx * 10), line)

    if closing_lines:
        canvas.setFillColor(colors.HexColor("#4a4a4a"))
        canvas.setFont(fonts.italic, 9.2)
        closing_y = footer_bottom_y + closing_h
        for idx, line in enumerate(closing_lines):
            canvas.drawString(frame.content_left, closing_y - (idx * 10), line)

def render_section_pattern_gallery_page(canvas: Canvas, payload: dict, *, frame: PageFrame) -> None:
    fonts = resolve_font_pack(payload.get("font_family"))
    page_height = frame.page_height

    headline = str(payload.get("headline") or f"{payload.get('title', 'Section')} Pattern Sneak Peek").strip()
    intro = str(payload.get("intro") or "").strip()

    title_block = draw_smart_headline(
        canvas,
        text=headline,
        font_name=fonts.bold,
        preferred_font_size=22.0,
        min_font_size=16.0,
        max_width=frame.content_width - 36,
        x=frame.content_left + 18,
        first_baseline_y=page_height - 108,
        align="left",
        max_lines=2,
        leading_multiplier=1.10,
        fill_color=colors.HexColor("#1f3c88"),
    )

    intro_text = intro or "A selected preview of patterns you will encounter in this section."
    intro_lines = _wrap_text(intro_text, 94)[:2]

    intro_y = title_block["bottom_y"] - 20
    intro_bottom_y = intro_y - ((len(intro_lines) - 1) * 13) if intro_lines else intro_y

    _draw_soft_panel(
        canvas,
        x=frame.content_left,
        y=intro_bottom_y - 18,
        w=frame.content_width,
        h=(page_height - 70) - (intro_bottom_y - 18),
        fill_hex="#f4f6fa",
    )

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont(fonts.bold, title_block["font_size"])
    for idx, line in enumerate(title_block["lines"]):
        canvas.drawString(frame.content_left + 18, (page_height - 108) - (idx * title_block["leading"]), line)

    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.setFont(fonts.regular, 10.5)
    for idx, line in enumerate(intro_lines):
        canvas.drawString(frame.content_left + 18, intro_y - (idx * 13), line)

    patterns = list(payload.get("patterns") or [])

    cols = 2
    rows = 4
    card_w = (frame.content_width - 24) / cols
    card_h = 118
    start_x = frame.content_left
    start_y = (intro_bottom_y - 18) - 30

    for idx, item in enumerate(patterns[: rows * cols]):
        row = idx // cols
        col = idx % cols
        x = start_x + col * card_w
        y_top = start_y - row * (card_h + 12)
        y = y_top - card_h

        canvas.setFillColor(colors.white)
        canvas.setStrokeColor(colors.HexColor("#d0d7e5"))
        canvas.setLineWidth(0.8)
        canvas.rect(x, y, card_w - 12, card_h, stroke=1, fill=1)

        preview_size = 56
        preview_x = x + 12
        preview_y = y + card_h - preview_size - 18
        _draw_mask_preview(
            canvas,
            x=preview_x,
            y=preview_y,
            size=preview_size,
            mask81=str(item.get("mask81") or ""),
        )

        text_x = preview_x + preview_size + 12
        text_y = y + card_h - 20

        canvas.setFillColor(colors.black)
        canvas.setFont(fonts.bold, 10.8)
        for line in _wrap_text(str(item.get("pattern_name") or item.get("pattern_id") or ""), 26)[:2]:
            canvas.drawString(text_x, text_y, line)
            text_y -= 13

        family_name = str(item.get("family_name") or "").strip()
        if family_name:
            canvas.setFont(fonts.regular, 9.2)
            canvas.setFillColor(colors.HexColor("#444444"))
            for line in _wrap_text(family_name, 28)[:2]:
                canvas.drawString(text_x, text_y, line)
                text_y -= 11

        page_refs_label = str(item.get("page_refs_label") or "").strip()
        if page_refs_label:
            canvas.setFont(fonts.regular, 8.8)
            canvas.setFillColor(colors.HexColor("#666666"))
            wrapped_refs = _wrap_text(page_refs_label, 34)[:2]

            footer_y = y + 28
            for line in wrapped_refs:
                canvas.drawString(text_x, footer_y, line)
                footer_y -= 10


def _range_label(a, b) -> str:
    if a in (None, "") and b in (None, ""):
        return "—"
    if a == b:
        return str(a)
    return f"{a} to {b}"