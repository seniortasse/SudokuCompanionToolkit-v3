from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from python.publishing.schemas.models import PublicationLayoutConfig

from .page_geometry import PageFrame, resolve_page_frame


_POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class PuzzleSlot:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class LayoutProfile:
    name: str
    trim_size: str
    page_size: Tuple[float, float]
    puzzles_per_page: int
    rows: int
    cols: int
    inner_margin: float
    outer_margin: float
    top_margin: float
    bottom_margin: float
    header_height: float
    footer_height: float
    gutter_x: float
    gutter_y: float
    slots: List[PuzzleSlot]
    frame: PageFrame


@dataclass(frozen=True)
class TemplateLayoutDefaults:
    name: str
    puzzles_per_page: int
    rows: int
    cols: int
    inner_margin_in: float
    outer_margin_in: float
    top_margin_in: float
    bottom_margin_in: float
    header_height_in: float
    footer_height_in: float
    gutter_x_in: float
    gutter_y_in: float


_TEMPLATE_LAYOUT_DEFAULTS: Dict[str, TemplateLayoutDefaults] = {
    "classic_1up_blackband": TemplateLayoutDefaults(
        name="classic_one_up_blackband",
        puzzles_per_page=1,
        rows=1,
        cols=1,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.0,
        gutter_y_in=0.0,
    ),
    "solution_1up_blackband": TemplateLayoutDefaults(
        name="solution_one_up_blackband",
        puzzles_per_page=1,
        rows=1,
        cols=1,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.0,
        gutter_y_in=0.0,
    ),
    "classic_2up_large": TemplateLayoutDefaults(
        name="classic_two_up",
        puzzles_per_page=2,
        rows=2,
        cols=1,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.30,
    ),
    "classic_2up_blackband": TemplateLayoutDefaults(
        name="classic_two_up_blackband",
        puzzles_per_page=2,
        rows=2,
        cols=1,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.30,
    ),
    "solution_2up_readable": TemplateLayoutDefaults(
        name="solution_two_up",
        puzzles_per_page=2,
        rows=2,
        cols=1,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.30,
    ),
    "solution_2up_blackband": TemplateLayoutDefaults(
        name="solution_two_up_blackband",
        puzzles_per_page=2,
        rows=2,
        cols=1,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.30,
    ),
    "classic_4up_clean": TemplateLayoutDefaults(
        name="classic_four_up",
        puzzles_per_page=4,
        rows=2,
        cols=2,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.25,
    ),
    "classic_4up_blackband": TemplateLayoutDefaults(
        name="classic_four_up_blackband",
        puzzles_per_page=4,
        rows=2,
        cols=2,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.25,
    ),
    "solution_4up_basic": TemplateLayoutDefaults(
        name="solution_four_up",
        puzzles_per_page=4,
        rows=2,
        cols=2,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.25,
    ),
    "solution_4up_blackband": TemplateLayoutDefaults(
        name="solution_four_up_blackband",
        puzzles_per_page=4,
        rows=2,
        cols=2,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.25,
    ),
    "solution_4up_readable": TemplateLayoutDefaults(
        name="solution_four_up_readable",
        puzzles_per_page=4,
        rows=2,
        cols=2,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.33,
        footer_height_in=0.33,
        gutter_x_in=0.25,
        gutter_y_in=0.25,
    ),
    "classic_6up_blackband": TemplateLayoutDefaults(
        name="classic_six_up_blackband",
        puzzles_per_page=6,
        rows=3,
        cols=2,
        inner_margin_in=0.80,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.28,
        footer_height_in=0.33,
        gutter_x_in=0.22,
        gutter_y_in=0.18,
    ),
    "solution_6up_blackband": TemplateLayoutDefaults(
        name="solution_six_up_blackband",
        puzzles_per_page=6,
        rows=3,
        cols=2,
        inner_margin_in=0.80,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.28,
        footer_height_in=0.33,
        gutter_x_in=0.22,
        gutter_y_in=0.18,
    ),
    "classic_12up_blackband": TemplateLayoutDefaults(
        name="classic_twelve_up_blackband",
        puzzles_per_page=12,
        rows=4,
        cols=3,
        inner_margin_in=0.75,
        outer_margin_in=0.50,
        top_margin_in=0.50,
        bottom_margin_in=0.50,
        header_height_in=0.22,
        footer_height_in=0.33,
        gutter_x_in=0.16,
        gutter_y_in=0.12,
    ),
    "solution_12up_blackband": TemplateLayoutDefaults(
        name="solution_twelve_up_blackband",
        puzzles_per_page=12,
        rows=4,
        cols=3,
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


def get_layout_profile(
    template_id: str,
    *,
    page_size: Tuple[float, float],
    trim_size: str,
    page_number: int = 1,
    mirror_margins: bool = True,
    layout_config: PublicationLayoutConfig | None = None,
    trim_page_size: Tuple[float, float] | None = None,
    bleed: float = 0.0,
) -> LayoutProfile:
    defaults = _resolve_template_defaults(template_id)
    cfg = layout_config or PublicationLayoutConfig()

    rows = int(cfg.rows if cfg.rows is not None else defaults.rows)
    cols = int(cfg.cols if cfg.cols is not None else defaults.cols)
    puzzles_per_page = int(
        cfg.puzzles_per_page if cfg.puzzles_per_page is not None else (rows * cols)
    )

    inner_margin = _points(cfg.inner_margin_in, defaults.inner_margin_in)
    outer_margin = _points(cfg.outer_margin_in, defaults.outer_margin_in)
    top_margin = _points(cfg.top_margin_in, defaults.top_margin_in)
    bottom_margin = _points(cfg.bottom_margin_in, defaults.bottom_margin_in)
    header_height = _points(cfg.header_height_in, defaults.header_height_in)
    footer_height = _points(cfg.footer_height_in, defaults.footer_height_in)
    gutter_x = _points(cfg.gutter_x_in, defaults.gutter_x_in)
    gutter_y = _points(cfg.gutter_y_in, defaults.gutter_y_in)

    frame = resolve_page_frame(
        page_size=page_size,
        page_number=page_number,
        mirror_margins=mirror_margins,
        inner_margin=inner_margin,
        outer_margin=outer_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        trim_size=trim_page_size,
        bleed=bleed,
    )

    usable_width = frame.content_width - ((cols - 1) * gutter_x)
    usable_height = (
        (frame.content_top - frame.content_bottom)
        - header_height
        - footer_height
        - ((rows - 1) * gutter_y)
    )

    slot_width = usable_width / float(cols)
    slot_height = usable_height / float(rows)

    slots: List[PuzzleSlot] = []
    for row in range(rows):
        for col in range(cols):
            x = frame.content_left + (col * (slot_width + gutter_x))
            y = (
                frame.content_top
                - header_height
                - ((row + 1) * slot_height)
                - (row * gutter_y)
            )
            slots.append(PuzzleSlot(x=x, y=y, width=slot_width, height=slot_height))

    return LayoutProfile(
        name=defaults.name,
        trim_size=trim_size,
        page_size=page_size,
        puzzles_per_page=puzzles_per_page,
        rows=rows,
        cols=cols,
        inner_margin=inner_margin,
        outer_margin=outer_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        header_height=header_height,
        footer_height=footer_height,
        gutter_x=gutter_x,
        gutter_y=gutter_y,
        slots=slots,
        frame=frame,
    )


def _resolve_template_defaults(template_id: str) -> TemplateLayoutDefaults:
    key = str(template_id).strip()
    if key in _TEMPLATE_LAYOUT_DEFAULTS:
        return _TEMPLATE_LAYOUT_DEFAULTS[key]
    return _TEMPLATE_LAYOUT_DEFAULTS["classic_4up_clean"]


def _points(value_in: float | None, default_in: float) -> float:
    return float(value_in if value_in is not None else default_in) * _POINTS_PER_INCH