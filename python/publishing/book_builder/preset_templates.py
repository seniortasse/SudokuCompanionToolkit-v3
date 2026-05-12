from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from python.publishing.book_builder.book_spec_loader import BookSectionSpec, BookSpec
from python.publishing.ids.id_policy import build_aisle_id, build_book_id, build_library_id

DW_DIFFICULTY_ORDER = [
    "easy",
    "medium",
    "hard",
    "expert",
    "genius",
]

DW_PRESET_RANGES = {
    "easy_to_medium": ("easy", "medium"),
    "easy_to_hard": ("easy", "hard"),
    "easy_to_expert": ("easy", "expert"),
    "easy_to_genius": ("easy", "genius"),
    "medium_to_hard": ("medium", "hard"),
    "medium_to_expert": ("medium", "expert"),
    "medium_to_genius": ("medium", "genius"),
    "hard_to_expert": ("hard", "expert"),
    "hard_to_genius": ("hard", "genius"),
    "expert_to_genius": ("expert", "genius"),
}


def _difficulty_slice(start: str, end: str) -> List[str]:
    start_idx = DW_DIFFICULTY_ORDER.index(start)
    end_idx = DW_DIFFICULTY_ORDER.index(end)
    if start_idx > end_idx:
        raise ValueError(f"Invalid DW preset range: {start} -> {end}")
    return DW_DIFFICULTY_ORDER[start_idx : end_idx + 1]


def _default_section_title(difficulty: str) -> str:
    return difficulty.capitalize()


def _default_section_subtitle(difficulty: str) -> str:
    mapping = {
        "easy": "Accessible warm-up puzzles",
        "medium": "A steady step up in challenge",
        "hard": "More demanding logic",
        "expert": "Advanced solving territory",
        "genius": "Top-end challenge",
    }
    return mapping.get(difficulty, "")


def build_dw_sections(
    *,
    preset_name: str,
    puzzles_per_section: int,
    section_overrides: Dict[str, Dict] | None = None,
) -> List[BookSectionSpec]:
    if preset_name not in DW_PRESET_RANGES:
        raise ValueError(f"Unsupported DW preset: {preset_name}")

    start, end = DW_PRESET_RANGES[preset_name]
    difficulties = _difficulty_slice(start, end)
    overrides = dict(section_overrides or {})

    sections: List[BookSectionSpec] = []
    for idx, difficulty in enumerate(difficulties, start=1):
        section_code = f"L{idx}"
        override = dict(overrides.get(section_code, {}))
        criteria = dict(override.get("criteria", {}))
        criteria.setdefault("puzzle_difficulty", difficulty)

        section = BookSectionSpec(
            section_code=section_code,
            title=str(override.get("title", _default_section_title(difficulty))),
            subtitle=str(override.get("subtitle", _default_section_subtitle(difficulty))),
            puzzle_count=int(override.get("puzzle_count", puzzles_per_section)),
            criteria=criteria,
            difficulty_label_hint=difficulty,
        )
        sections.append(section)

    return sections


def build_dw_book_spec(
    *,
    library_short: str,
    aisle_short: str,
    book_number: int,
    preset_name: str,
    puzzles_per_section: int,
    trim_size: str = "8.5x11",
    puzzles_per_page: int = 1,
    page_layout_profile: str = "classic_single",
    solution_section_policy: str = "appendix",
    cover_theme: str = "classic",
    layout_type: str = "classic9x9",
    grid_size: int = 9,
    search_tags: List[str] | None = None,
    publication_status: str = "draft",
    global_filters: Dict | None = None,
    ordering_policy: Dict | None = None,
    reuse_policy: str = "book_exclusive",
    title: str | None = None,
    subtitle: str = "",
    series_name: str = "",
    volume_number: int | None = None,
    isbn: str | None = None,
    description: str = "",
    target_audience: str = "general",
    section_overrides: Dict[str, Dict] | None = None,
) -> BookSpec:
    library_id = build_library_id(library_short)
    aisle_id = build_aisle_id(aisle_short)
    book_id = build_book_id(library_short, aisle_short, book_number)

    sections = build_dw_sections(
        preset_name=preset_name,
        puzzles_per_section=puzzles_per_section,
        section_overrides=section_overrides,
    )

    if title is None:
        title = f"{preset_name.replace('_', ' ').title()} Sudoku"

    effective_global_filters = dict(global_filters or {})
    effective_global_filters.setdefault("layout_type", layout_type)
    effective_global_filters.setdefault("is_unique", True)
    effective_global_filters.setdefault("is_human_solvable", True)
    effective_global_filters.setdefault("candidate_status_in", ["available"])

    effective_ordering = dict(ordering_policy or {})
    effective_ordering.setdefault("within_section", "weight_asc")
    effective_ordering.setdefault("tie_breakers", ["technique_count", "clue_count", "generation_seed"])

    return BookSpec(
        book_id=book_id,
        library_id=library_id,
        aisle_id=aisle_id,
        title=title,
        subtitle=subtitle,
        series_name=series_name,
        volume_number=volume_number,
        isbn=isbn,
        description=description,
        target_audience=target_audience,
        trim_size=trim_size,
        puzzles_per_page=puzzles_per_page,
        page_layout_profile=page_layout_profile,
        solution_section_policy=solution_section_policy,
        cover_theme=cover_theme,
        layout_type=layout_type,
        grid_size=grid_size,
        search_tags=list(search_tags or []),
        publication_status=publication_status,
        global_filters=effective_global_filters,
        ordering_policy=effective_ordering,
        reuse_policy=reuse_policy,
        sections=sections,
    )