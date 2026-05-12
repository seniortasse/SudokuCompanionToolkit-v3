from __future__ import annotations

from typing import Dict, List

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from python.publishing.i18n.strings import tr

from python.publishing.schemas.models import PageBlock, PublicationLayoutConfig, PuzzleRecord
from python.publishing.schemas.page_types import (
    BLANK_PAGE,
    FEATURES_PAGE,
    PROMO_PAGE,
    PUZZLE_PAGE,
    RULES_PAGE,
    SECTION_HIGHLIGHTS_PAGE,
    SECTION_OPENER_PAGE,
    SECTION_PATTERN_GALLERY_PAGE,
    SOLUTION_PAGE,
    TITLE_PAGE,
    TOC_PAGE,
    TUTORIAL_PAGE,
    WARMUP_PAGE,
    WELCOME_PAGE,
)

from .front_matter_renderer import render_title_page
from .layout_profiles import get_layout_profile
from .page_geometry import resolve_page_frame
from .puzzle_page_renderer import render_puzzle_page
from .render_models import PublicationRenderContext, RenderSection
from .rich_front_matter_renderer import (
    render_features_page,
    render_promo_page,
    render_rules_page,
    render_toc_page,
    render_tutorial_page,
    render_warmup_page,
    render_welcome_page,
)
from .section_divider_renderer import render_section_divider_page
from .section_prelude_renderer import (
    render_section_highlights_page,
    render_section_pattern_gallery_page,
)
from .solution_page_renderer import render_solution_page


_POINTS_PER_INCH = 72.0


def render_page_block(
    canvas: Canvas,
    *,
    block: PageBlock,
    context: PublicationRenderContext,
) -> None:
    page_number = int(block.physical_page_number or block.page_index or 1)
    mirror_margins = bool(context.publication_manifest.get("mirror_margins", False))
    layout_config = _publication_layout_config(context)

    frame = resolve_page_frame(
        page_size=canvas._pagesize,
        page_number=page_number,
        mirror_margins=mirror_margins,
        inner_margin=_points(layout_config.inner_margin_in, 0.75),
        outer_margin=_points(layout_config.outer_margin_in, 0.50),
        top_margin=_points(layout_config.top_margin_in, 0.50),
        bottom_margin=_points(layout_config.bottom_margin_in, 0.50),
        trim_size=_trim_size_points(context),
        bleed=_render_bleed_points(context),
    )

    if isinstance(block.payload, dict):
        block.payload.setdefault("language", str(layout_config.language or "en"))
        block.payload.setdefault("font_family", str(layout_config.font_family or "helvetica"))

    if block.page_type == TITLE_PAGE:
        render_title_page(
            canvas,
            context.render_model,
            frame=frame,
            payload=block.payload,
            publication_manifest=context.publication_manifest,
            auto_advance=False,
        )
        return

    if block.page_type == WELCOME_PAGE:
        render_welcome_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == FEATURES_PAGE:
        render_features_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == TOC_PAGE:
        render_toc_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == RULES_PAGE:
        render_rules_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == TUTORIAL_PAGE:
        render_tutorial_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == WARMUP_PAGE:
        render_warmup_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == PROMO_PAGE:
        render_promo_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == SECTION_OPENER_PAGE:
        section = _find_section(context, block.section_id)
        render_section_divider_page(
            canvas,
            section,
            payload=block.payload,
            publication_manifest=context.publication_manifest,
            frame=frame,
            auto_advance=False,
        )
        return

    if block.page_type == SECTION_HIGHLIGHTS_PAGE:
        render_section_highlights_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == SECTION_PATTERN_GALLERY_PAGE:
        render_section_pattern_gallery_page(canvas, block.payload, frame=frame)
        return

    if block.page_type == PUZZLE_PAGE:
        puzzles = _resolve_puzzles(context, block.payload.get("puzzle_ids", []))
        layout_profile = _resolve_layout_profile(context, block, canvas)
        page_title = str(block.payload.get("page_title") or _default_section_page_title(block))
        render_puzzle_page(
            canvas,
            puzzles=puzzles,
            layout_profile=layout_profile,
            page_title=page_title,
            show_solution=False,
            layout_config=_publication_layout_config(context),
        )
        return

    if block.page_type == SOLUTION_PAGE:
        puzzles = _resolve_puzzles(context, block.payload.get("puzzle_ids", []))
        layout_profile = _resolve_layout_profile(context, block, canvas)
        layout_config = _solution_publication_layout_config(context)
        language = str(
            block.payload.get("language")
            or layout_config.language
            or "en"
        )
        page_title = str(block.payload.get("page_title") or tr("solutions", language))
        render_solution_page(
            canvas,
            puzzles=puzzles,
            layout_profile=layout_profile,
            page_title=page_title,
            layout_config=layout_config,
        )
        return

    if block.page_type == "SOLUTION_SECTION_OPENER_PAGE":
        solution_section_config = dict(
            context.publication_manifest.get("solution_section_config") or {}
        )
        opener_payload = dict(solution_section_config)
        if isinstance(block.payload, dict):
            opener_payload.update(block.payload)

        render_simple_solution_section_opener(canvas, opener_payload, frame=frame)
        return

    if block.page_type == BLANK_PAGE:
        return

    raise ValueError(f"Unsupported page block type in renderer: {block.page_type}")


def _resolve_layout_profile(
    context: PublicationRenderContext,
    block: PageBlock,
    canvas: Canvas,
):
    layout_config = (
        _solution_publication_layout_config(context)
        if block.page_type == SOLUTION_PAGE
        else _publication_layout_config(context)
    )

    return get_layout_profile(
        block.template_id,
        page_size=canvas._pagesize,
        trim_size=str(context.publication_manifest.get("trim_size_label") or context.render_model.book_manifest.trim_size),
        page_number=int(block.physical_page_number or block.page_index or 1),
        mirror_margins=bool(context.publication_manifest.get("mirror_margins", False)),
        layout_config=layout_config,
        trim_page_size=_trim_size_points(context),
        bleed=_render_bleed_points(context),
    )


def _publication_layout_config(context: PublicationRenderContext) -> PublicationLayoutConfig:
    return PublicationLayoutConfig.from_dict(context.publication_manifest.get("layout_config"))


def _solution_publication_layout_config(context: PublicationRenderContext) -> PublicationLayoutConfig:
    cfg = PublicationLayoutConfig.from_dict(context.publication_manifest.get("layout_config"))

    if getattr(cfg, "solution_puzzles_per_page", None) is not None:
        cfg.puzzles_per_page = int(cfg.solution_puzzles_per_page)
    if getattr(cfg, "solution_rows", None) is not None:
        cfg.rows = int(cfg.solution_rows)
    if getattr(cfg, "solution_cols", None) is not None:
        cfg.cols = int(cfg.solution_cols)

    return cfg


def render_simple_solution_section_opener(canvas: Canvas, payload: dict, *, frame) -> None:
    headline = str(payload.get("headline") or "Solutions Included").strip()
    kicker = str(payload.get("kicker") or "Answer key").strip()
    body = str(payload.get("body") or "Use these pages to check your completed grids.").strip()
    usage_tips = [str(x).strip() for x in list(payload.get("usage_tips") or []) if str(x).strip()]
    solution_puzzles_per_page = payload.get("solution_puzzles_per_page")

    answer_key_layout_label = str(
        payload.get("answer_key_layout_label") or "Answer key layout"
    ).strip()
    solutions_per_page_label = str(
        payload.get("solutions_per_page_label") or "solutions per page"
    ).strip()
    usage_tips_title = str(
        payload.get("usage_tips_title") or "How to use these pages"
    ).strip()
    footer_note = str(
        payload.get("footer_note")
        or "Check gently. Learn from the answer key. Then keep solving."
    ).strip()

    font_family = str(payload.get("font_family") or "arial").strip().lower()
    font_regular = _embedded_font_name(font_family, role="regular")
    font_bold = _embedded_font_name(font_family, role="bold")
    font_italic = _embedded_font_name(font_family, role="italic")

    canvas.saveState()

    # Page background.
    canvas.setFillColor(colors.HexColor("#06251F"))
    canvas.rect(frame.trim_left, frame.trim_bottom, frame.trim_width, frame.trim_height, stroke=0, fill=1)

    # Gold frame.
    canvas.setStrokeColor(colors.HexColor("#E6C461"))
    canvas.setLineWidth(1.6)
    inset = 24
    canvas.rect(
        frame.trim_left + inset,
        frame.trim_bottom + inset,
        frame.trim_width - 2 * inset,
        frame.trim_height - 2 * inset,
        stroke=1,
        fill=0,
    )

    # Main content.
    x = frame.content_left
    y = frame.content_top - 92

    canvas.setFillColor(colors.HexColor("#E6C461"))
    canvas.setFont(font_bold, 32)
    canvas.drawString(x, y, headline)

    y -= 34
    canvas.setFillColor(colors.HexColor("#F7F4E8"))
    canvas.setFont(font_bold, 14)
    canvas.drawString(x, y, kicker)

    y -= 42
    canvas.setFont(font_regular, 12.5)
    for line in _wrap_text(body, max_chars=82):
        canvas.drawString(x, y, line)
        y -= 18

    if solution_puzzles_per_page:
        y -= 18
        canvas.setFillColor(colors.HexColor("#E6C461"))
        canvas.setFont(font_bold, 13)
        canvas.drawString(
            x,
            y,
            f"{answer_key_layout_label}: {int(solution_puzzles_per_page)} {solutions_per_page_label}",
        )
        y -= 28

    if usage_tips:
        y -= 10
        canvas.setFillColor(colors.HexColor("#F7F4E8"))
        canvas.setFont(font_bold, 13)
        canvas.drawString(x, y, usage_tips_title)
        y -= 24

        canvas.setFont(font_regular, 11.5)
        for tip in usage_tips:
            wrapped = _wrap_text(tip, max_chars=78)
            if not wrapped:
                continue
            canvas.drawString(x + 12, y, f"• {wrapped[0]}")
            y -= 16
            for continuation in wrapped[1:]:
                canvas.drawString(x + 26, y, continuation)
                y -= 16
            y -= 4

    canvas.setFillColor(colors.HexColor("#E6C461"))
    canvas.setFont(font_italic, 10)
    canvas.drawCentredString(
        frame.trim_left + frame.trim_width / 2,
        frame.footer_baseline_y,
        footer_note,
    )

    canvas.restoreState()


def _embedded_font_name(font_family: str, *, role: str) -> str:
    """
    Pick an already-registered embedded font instead of ReportLab base fonts.

    The old solution opener used Helvetica / Helvetica-Bold / Helvetica-Oblique.
    Those PDF base fonts are not reliably embedded and can trigger KDP warnings.
    This helper prefers the registered Arial names used by the rest of the book.
    """
    registered = set(pdfmetrics.getRegisteredFontNames())

    role = str(role or "regular").strip().lower()
    family = str(font_family or "arial").strip().lower()

    if family in {"arial", "sans", "sans-serif"}:
        candidates_by_role = {
            "regular": [
                "Arial",
                "ArialMT",
                "Arial-Regular",
                "ArialUnicode",
                "DejaVuSans",
            ],
            "bold": [
                "Arial-Bold",
                "Arial-BoldMT",
                "ArialBold",
                "Arial-BoldItalic",
                "DejaVuSans-Bold",
            ],
            "italic": [
                "Arial-Italic",
                "Arial-ItalicMT",
                "Arial-Oblique",
                "ArialItalic",
                "DejaVuSans-Oblique",
                "DejaVuSans-Italic",
            ],
        }
    else:
        candidates_by_role = {
            "regular": [
                font_family,
                f"{font_family}-Regular",
                "Arial",
                "ArialMT",
                "DejaVuSans",
            ],
            "bold": [
                f"{font_family}-Bold",
                f"{font_family}Bold",
                "Arial-Bold",
                "Arial-BoldMT",
                "DejaVuSans-Bold",
            ],
            "italic": [
                f"{font_family}-Italic",
                f"{font_family}-Oblique",
                f"{font_family}Italic",
                "Arial-Italic",
                "Arial-ItalicMT",
                "DejaVuSans-Oblique",
                "DejaVuSans-Italic",
            ],
        }

    for candidate in candidates_by_role.get(role, candidates_by_role["regular"]):
        if candidate in registered:
            return candidate

    # Last resort: do not crash the build. If this fallback is reached,
    # the PDF may still contain a base font, so it is worth checking the
    # global font registration code.
    fallback_by_role = {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "italic": "Helvetica-Oblique",
    }
    return fallback_by_role.get(role, "Helvetica")


def _wrap_text(text: str, *, max_chars: int) -> List[str]:
    words = str(text or "").split()
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def _points(value_in: float | None, default_in: float) -> float:
    return float(value_in if value_in is not None else default_in) * _POINTS_PER_INCH

def _trim_size_points(context: PublicationRenderContext) -> tuple[float, float]:
    explicit = context.publication_manifest.get("_render_trim_size_pt")
    if explicit:
        return (float(explicit[0]), float(explicit[1]))

    trim = dict(context.publication_manifest.get("trim_size") or {})
    return (
        float(trim.get("width_in") or context.render_model.book_manifest.trim_size.split("x")[0]) * _POINTS_PER_INCH,
        float(trim.get("height_in") or 11.0) * _POINTS_PER_INCH,
    )


def _render_bleed_points(context: PublicationRenderContext) -> float:
    return float(context.publication_manifest.get("_render_bleed_pt") or 0.0)

def _find_section(context: PublicationRenderContext, section_id: str | None) -> RenderSection:
    if not section_id:
        raise ValueError("Section page block is missing section_id")

    for section in context.render_model.sections:
        if section.section_manifest.section_id == section_id:
            return section

    raise KeyError(f"Unknown section_id '{section_id}' in publication interior plan")


def _build_puzzle_index(context: PublicationRenderContext) -> Dict[str, PuzzleRecord]:
    out: Dict[str, PuzzleRecord] = {}
    for section in context.render_model.sections:
        for puzzle in section.puzzles:
            out[puzzle.puzzle_uid] = puzzle
    return out


def _resolve_puzzles(context: PublicationRenderContext, puzzle_ids: List[str]) -> List[PuzzleRecord]:
    puzzle_by_id = _build_puzzle_index(context)
    missing = [puzzle_id for puzzle_id in puzzle_ids if puzzle_id not in puzzle_by_id]
    if missing:
        raise KeyError(f"Interior plan references missing puzzles: {missing}")
    return [puzzle_by_id[puzzle_id] for puzzle_id in puzzle_ids]


def _default_section_page_title(block: PageBlock) -> str:
    section_code = block.payload.get("section_code")
    section_title = block.payload.get("section_title")
    if section_code and section_title:
        return f"{section_code} - {section_title}"
    if section_title:
        return str(section_title)
    return "Sudoku Section"