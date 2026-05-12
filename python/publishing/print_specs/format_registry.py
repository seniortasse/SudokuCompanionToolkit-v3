from __future__ import annotations

from typing import Dict, List

from python.publishing.schemas.models import PrintFormatSpec


_FORMATS: Dict[str, PrintFormatSpec] = {
    "amazon_kdp_paperback_8_5x11_bw": PrintFormatSpec(
        format_id="amazon_kdp_paperback_8_5x11_bw",
        vendor="amazon_kdp",
        binding_type="paperback",
        trim_width_in=8.5,
        trim_height_in=11.0,
        bleed_in=0.125,
        safe_margin_in=0.25,
        inside_margin_in=0.5,
        outside_margin_in=0.25,
        top_margin_in=0.25,
        bottom_margin_in=0.375,
        supports_spine=True,
        supports_isbn=True,
        paper_options=["white_bw", "cream_bw"],
        color_options=["black_white"],
        description="Amazon KDP paperback preset for 8.5 x 11 inch black-and-white interiors.",
    ),
    "amazon_kdp_paperback_8_5x11_color": PrintFormatSpec(
        format_id="amazon_kdp_paperback_8_5x11_color",
        vendor="amazon_kdp",
        binding_type="paperback",
        trim_width_in=8.5,
        trim_height_in=11.0,
        bleed_in=0.125,
        safe_margin_in=0.25,
        inside_margin_in=0.5,
        outside_margin_in=0.25,
        top_margin_in=0.25,
        bottom_margin_in=0.375,
        supports_spine=True,
        supports_isbn=True,
        paper_options=["white_color"],
        color_options=["premium_color", "standard_color"],
        description="Amazon KDP paperback preset for 8.5 x 11 inch color interiors.",
    ),
    "amazon_kdp_paperback_6x9_bw": PrintFormatSpec(
        format_id="amazon_kdp_paperback_6x9_bw",
        vendor="amazon_kdp",
        binding_type="paperback",
        trim_width_in=6.0,
        trim_height_in=9.0,
        bleed_in=0.125,
        safe_margin_in=0.25,
        inside_margin_in=0.375,
        outside_margin_in=0.25,
        top_margin_in=0.25,
        bottom_margin_in=0.375,
        supports_spine=True,
        supports_isbn=True,
        paper_options=["white_bw", "cream_bw"],
        color_options=["black_white"],
        description="Amazon KDP paperback preset for 6 x 9 inch black-and-white interiors.",
    ),
}


def get_print_format_spec(format_id: str) -> PrintFormatSpec:
    try:
        return _FORMATS[format_id]
    except KeyError as exc:
        known = ", ".join(sorted(_FORMATS.keys()))
        raise KeyError(f"Unknown print format '{format_id}'. Known formats: {known}") from exc


def list_print_format_specs() -> List[PrintFormatSpec]:
    return [_FORMATS[key] for key in sorted(_FORMATS.keys())]