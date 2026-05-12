from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PageFrame:
    # Physical PDF page / MediaBox.
    page_width: float
    page_height: float

    # Final trimmed page.
    trim_width: float
    trim_height: float
    trim_left: float
    trim_bottom: float
    trim_right: float
    trim_top: float

    # Bleed around trim.
    bleed: float
    outside_bleed: float
    top_bleed: float
    bottom_bleed: float

    page_number: int
    is_even_page: bool
    mirror_margins: bool

    inner_margin: float
    outer_margin: float
    top_margin: float
    bottom_margin: float
    left_margin: float
    right_margin: float

    content_left: float
    content_right: float
    content_width: float
    content_top: float
    content_bottom: float
    footer_baseline_y: float


def resolve_page_frame(
    *,
    page_size: Tuple[float, float],
    page_number: int,
    mirror_margins: bool,
    inner_margin: float = 54.0,
    outer_margin: float = 36.0,
    top_margin: float = 36.0,
    bottom_margin: float = 36.0,
    footer_baseline_y: float = 31.0,
    trim_size: Tuple[float, float] | None = None,
    bleed: float = 0.0,
) -> PageFrame:
    page_width, page_height = page_size
    page_number = max(1, int(page_number or 1))
    is_even_page = (page_number % 2 == 0)
    bleed = max(0.0, float(bleed or 0.0))

    if trim_size is None:
        trim_width, trim_height = page_width, page_height
    else:
        trim_width, trim_height = trim_size

    trim_width = float(trim_width)
    trim_height = float(trim_height)

    # KDP interior bleed convention:
    # PDF width  = trim width + outside bleed only.
    # PDF height = trim height + top bleed + bottom bleed.
    # Odd pages bleed on the right/outside edge.
    # Even pages bleed on the left/outside edge.
    #
    # No bleed:
    #   trim_left = 0
    #   trim_bottom = 0
    #
    # Bleed:
    #   odd page:  trim_left = 0
    #   even page: trim_left = bleed
    #   all pages: trim_bottom = bleed
    if bleed > 0:
        trim_left = bleed if is_even_page else 0.0
        trim_bottom = bleed
        outside_bleed = bleed
        top_bleed = bleed
        bottom_bleed = bleed
    else:
        trim_left = 0.0
        trim_bottom = 0.0
        outside_bleed = 0.0
        top_bleed = 0.0
        bottom_bleed = 0.0

    trim_right = trim_left + trim_width
    trim_top = trim_bottom + trim_height

    if mirror_margins:
        if is_even_page:
            left_margin = outer_margin
            right_margin = inner_margin
        else:
            left_margin = inner_margin
            right_margin = outer_margin
    else:
        left_margin = outer_margin
        right_margin = outer_margin

    content_left = trim_left + left_margin
    content_right = trim_right - right_margin
    content_top = trim_top - top_margin
    content_bottom = trim_bottom + bottom_margin

    return PageFrame(
        page_width=page_width,
        page_height=page_height,
        trim_width=trim_width,
        trim_height=trim_height,
        trim_left=trim_left,
        trim_bottom=trim_bottom,
        trim_right=trim_right,
        trim_top=trim_top,
        bleed=bleed,
        outside_bleed=outside_bleed,
        top_bleed=top_bleed,
        bottom_bleed=bottom_bleed,
        page_number=page_number,
        is_even_page=is_even_page,
        mirror_margins=mirror_margins,
        inner_margin=inner_margin,
        outer_margin=outer_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
        content_left=content_left,
        content_right=content_right,
        content_width=content_right - content_left,
        content_top=content_top,
        content_bottom=content_bottom,
        # Keep footer text inside KDP's safe text area.
        # 0.42 in = 30.24 pt above the trim bottom.
        #
        # IMPORTANT:
        # In bleed mode trim_bottom is 0.125 in = 9 pt, so the footer baseline is
        # naturally higher on the PDF MediaBox than it is in the no-bleed file.
        # Any bottom note that sits above the footer must therefore use this value
        # or a helper derived from it, never a raw absolute y such as 42, 52, 54, or 56.
        footer_baseline_y=trim_bottom + max(float(footer_baseline_y or 0.0), 0.42 * 72.0),
    )