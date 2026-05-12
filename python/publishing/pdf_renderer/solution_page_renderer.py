from __future__ import annotations

from typing import List

from python.publishing.schemas.models import PublicationLayoutConfig, PuzzleRecord
from reportlab.pdfgen.canvas import Canvas

from .layout_profiles import LayoutProfile
from .puzzle_page_renderer import render_puzzle_page


def render_solution_page(
    canvas: Canvas,
    *,
    puzzles: List[PuzzleRecord],
    layout_profile: LayoutProfile,
    page_title: str,
    layout_config: PublicationLayoutConfig | None = None,
) -> None:
    render_puzzle_page(
        canvas,
        puzzles=puzzles,
        layout_profile=layout_profile,
        page_title=page_title,
        show_solution=True,
        layout_config=layout_config,
    )