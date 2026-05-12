from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LayoutPreset:
    preset_id: str
    puzzles_per_page: int
    rows: int
    cols: int
    puzzle_page_template: str
    solution_page_template: str
    inner_margin_in: float
    outer_margin_in: float
    top_margin_in: float
    bottom_margin_in: float
    header_height_in: float
    footer_height_in: float
    gutter_x_in: float
    gutter_y_in: float


_LAYOUT_PRESETS: Dict[str, LayoutPreset] = {
    "1up": LayoutPreset(
        preset_id="1up",
        puzzles_per_page=1,
        rows=1,
        cols=1,
        puzzle_page_template="classic_1up_blackband",
        solution_page_template="solution_1up_blackband",
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.0,
        gutter_y_in=0.0,
    ),
    "2up": LayoutPreset(
        preset_id="2up",
        puzzles_per_page=2,
        rows=2,
        cols=1,
        puzzle_page_template="classic_2up_blackband",
        solution_page_template="solution_2up_blackband",
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.30,
    ),
    "4up": LayoutPreset(
        preset_id="4up",
        puzzles_per_page=4,
        rows=2,
        cols=2,
        puzzle_page_template="classic_4up_blackband",
        solution_page_template="solution_4up_blackband",
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.25,
    ),
    "6up": LayoutPreset(
        preset_id="6up",
        puzzles_per_page=6,
        rows=3,
        cols=2,
        puzzle_page_template="classic_6up_blackband",
        solution_page_template="solution_6up_blackband",
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.28,
        footer_height_in=0.33,
        gutter_x_in=0.22,
        gutter_y_in=0.18,
    ),
    "12up": LayoutPreset(
        preset_id="12up",
        puzzles_per_page=12,
        rows=4,
        cols=3,
        puzzle_page_template="classic_12up_blackband",
        solution_page_template="solution_12up_blackband",
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.22,
        footer_height_in=0.33,
        gutter_x_in=0.16,
        gutter_y_in=0.12,
    ),
}


_COUNT_TO_PRESET_ID = {
    1: "1up",
    2: "2up",
    4: "4up",
    6: "6up",
    12: "12up",
}


def normalize_layout_preset_id(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "1": "1up",
        "1up": "1up",
        "1-up": "1up",
        "2": "2up",
        "2up": "2up",
        "2-up": "2up",
        "4": "4up",
        "4up": "4up",
        "4-up": "4up",
        "6": "6up",
        "6up": "6up",
        "6-up": "6up",
        "12": "12up",
        "12up": "12up",
        "12-up": "12up",
    }
    return aliases.get(raw, raw)


def get_layout_preset(preset_id: str) -> LayoutPreset:
    key = normalize_layout_preset_id(preset_id)
    if key not in _LAYOUT_PRESETS:
        raise KeyError(
            f"Unknown layout preset '{preset_id}'. "
            f"Supported presets: {', '.join(sorted(_LAYOUT_PRESETS.keys()))}"
        )
    return _LAYOUT_PRESETS[key]


def infer_layout_preset_id_from_count(puzzles_per_page: int | None) -> Optional[str]:
    if puzzles_per_page is None:
        return None
    return _COUNT_TO_PRESET_ID.get(int(puzzles_per_page))


def resolve_layout_preset(
    *,
    layout_preset_id: str | None = None,
    puzzles_per_page: int | None = None,
) -> Optional[LayoutPreset]:
    if layout_preset_id:
        return get_layout_preset(layout_preset_id)

    inferred = infer_layout_preset_id_from_count(puzzles_per_page)
    if inferred:
        return get_layout_preset(inferred)

    return None


def list_layout_preset_ids() -> list[str]:
    return sorted(_LAYOUT_PRESETS.keys())