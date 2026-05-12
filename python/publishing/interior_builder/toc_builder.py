from __future__ import annotations

from typing import Dict, List, Set, Tuple

from python.publishing.i18n.strings import format_of_total, tr, translate_difficulty_label

from python.publishing.schemas.models import InteriorPlan
from python.publishing.schemas.page_types import (
    FEATURES_PAGE,
    PUZZLE_PAGE,
    RULES_PAGE,
    SECTION_HIGHLIGHTS_PAGE,
    SECTION_OPENER_PAGE,
    SECTION_PATTERN_GALLERY_PAGE,
    SOLUTION_PAGE,
    TOC_PAGE,
    TUTORIAL_PAGE,
    WARMUP_PAGE,
    WELCOME_PAGE,
)


def build_toc_entries(plan: InteriorPlan) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    seen_section_specials: Set[Tuple[str, str]] = set()

    for block in plan.page_blocks:
        if block.page_type == WELCOME_PAGE:
            entries.append(
                {
                    "title": _welcome_toc_title(block),
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == FEATURES_PAGE:
            entries.append(
                {
                    "title": _features_toc_title(block),
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == TOC_PAGE:
            entries.append(
                {
                    "title": tr("contents", _payload_language(block)),
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == RULES_PAGE:
            entries.append(
                {
                    "title": _series_toc_title(
                        block,
                        editorial_key="rules",
                        default_eyebrow=tr("sudoku_rules", _payload_language(block)),
                    ),
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == TUTORIAL_PAGE:
            entries.append(
                {
                    "title": _series_toc_title(
                        block,
                        editorial_key="tutorial",
                        default_eyebrow=tr("tutorial", _payload_language(block)),
                    ),
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == WARMUP_PAGE:
            entries.append(
                {
                    "title": _series_toc_title(
                        block,
                        editorial_key="warmup",
                        default_eyebrow=tr("warmup", _payload_language(block)),
                    ),
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == SECTION_OPENER_PAGE:
            section_code = str(block.payload.get("section_code", "")).strip()
            section_title = str(block.payload.get("title", "Section")).strip()
            if section_code:
                display = f"{section_code} - {section_title}"
            else:
                display = section_title
            entries.append(
                {
                    "title": display,
                    "page_index": _toc_page_number(block),
                }
            )

        elif block.page_type == SECTION_HIGHLIGHTS_PAGE:
            section_key = (str(block.section_id or ""), "highlights")
            if section_key not in seen_section_specials:
                seen_section_specials.add(section_key)
                section_title = str(block.payload.get("title", "Section")).strip()
                entries.append(
                    {
                        "title": f"{section_title} • {tr('section_highlights', _payload_language(block))}",
                        "page_index": _toc_page_number(block),
                    }
                )

        elif block.page_type == SECTION_PATTERN_GALLERY_PAGE:
            section_key = (str(block.section_id or ""), "gallery")
            if section_key not in seen_section_specials:
                seen_section_specials.add(section_key)
                section_title = str(block.payload.get("title", "Section")).strip()
                headline = str(block.payload.get("headline") or "").strip()
                display = headline if headline else f"{section_title} Pattern Sneak Peek"
                entries.append(
                    {
                        "title": display,
                        "page_index": _toc_page_number(block),
                    }
                )

        elif block.page_type == "SOLUTION_SECTION_OPENER_PAGE":
            payload = _payload(block)
            solution_title = str(payload.get("toc_title") or payload.get("headline") or "").strip()
            if not solution_title:
                solution_title = tr("solutions", _payload_language(block))
            if not any(e["title"] == solution_title for e in entries):
                entries.append(
                    {
                        "title": solution_title,
                        "page_index": _toc_page_number(block),
                    }
                )

        elif block.page_type == SOLUTION_PAGE:
            solution_title = tr("solutions", _payload_language(block))
            if not any(e["title"] == solution_title for e in entries):
                entries.append(
                    {
                        "title": solution_title,
                        "page_index": _toc_page_number(block),
                    }
                )

        elif block.page_type == PUZZLE_PAGE:
            continue

    return entries


def _payload(block) -> dict:
    return dict(getattr(block, "payload", {}) or {})


def _editorial_copy_section(block, key: str) -> dict:
    payload = _payload(block)
    editorial_copy = dict(payload.get("editorial_copy") or {})
    return dict(editorial_copy.get(key) or {})


def _payload_language(block) -> str:
    payload = _payload(block)
    return str(payload.get("language") or "en")


def _welcome_toc_title(block) -> str:
    payload = _payload(block)
    editorial = _editorial_copy_section(block, "welcome")
    language = _payload_language(block)

    explicit = str(payload.get("toc_title") or "").strip()
    if explicit:
        return explicit

    band_label = str(
        payload.get("band_label")
        or editorial.get("eyebrow")
        or tr("welcome", language)
    ).strip()
    return band_label or tr("welcome", language)


def _features_toc_title(block) -> str:
    payload = _payload(block)
    language = _payload_language(block)

    explicit = str(payload.get("toc_title") or "").strip()
    if explicit:
        return explicit

    publication_metadata = dict(payload.get("publication_metadata") or {})
    raw_audience = str(publication_metadata.get("audience") or "Adults").strip()
    audience = tr("audience_adults", language) if raw_audience.lower() == "adults" else raw_audience
    companion_features = tr("companion_features", language)

    if audience:
        return f"{audience} • {companion_features}"
    return companion_features


def _series_toc_title(block, *, editorial_key: str, default_eyebrow: str) -> str:
    payload = _payload(block)
    language = _payload_language(block)

    explicit = str(payload.get("toc_title") or "").strip()
    if explicit:
        return explicit

    page_index = int(payload.get("page_occurrence_index") or 1)
    page_total = int(payload.get("page_occurrence_total") or 1)

    editorial = _editorial_copy_section(block, editorial_key)
    page_cfg = dict(editorial.get(f"page{page_index}") or editorial.get("page1") or {})
    eyebrow = str(page_cfg.get("eyebrow") or default_eyebrow).strip() or default_eyebrow

    if page_total > 1:
        return f"{eyebrow} • {format_of_total(page_index, page_total, language)}"
    return eyebrow


def _toc_page_number(block) -> int:
    if block.logical_page_number not in (None, 0):
        return int(block.logical_page_number)
    if block.physical_page_number not in (None, 0):
        return int(block.physical_page_number)
    return int(block.page_index or 0)