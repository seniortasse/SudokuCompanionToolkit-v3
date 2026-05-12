from __future__ import annotations

from typing import List

from python.publishing.pdf_renderer.layout_validation import validate_layout_geometry
from python.publishing.pdf_renderer.typography import (
    is_supported_digit_preset,
    is_supported_font_family,
)
from python.publishing.print_specs.channel_registry import get_channel_preset
from python.publishing.schemas.models import PrintFormatSpec, PublicationSpec


def validate_print_format_spec(spec: PrintFormatSpec) -> List[str]:
    errors: List[str] = []

    if spec.trim_width_in <= 0:
        errors.append("trim_width_in must be > 0")
    if spec.trim_height_in <= 0:
        errors.append("trim_height_in must be > 0")
    if spec.bleed_in < 0:
        errors.append("bleed_in must be >= 0")
    if spec.safe_margin_in < 0:
        errors.append("safe_margin_in must be >= 0")
    if spec.inside_margin_in < 0 or spec.outside_margin_in < 0:
        errors.append("inside/outside margins must be >= 0")
    if spec.top_margin_in < 0 or spec.bottom_margin_in < 0:
        errors.append("top/bottom margins must be >= 0")

    return errors


def validate_publication_spec(spec: PublicationSpec, format_spec: PrintFormatSpec) -> List[str]:
    errors: List[str] = []

    if not spec.publication_id.strip():
        errors.append("publication_id must not be blank")
    if not spec.book_id.strip():
        errors.append("book_id must not be blank")
    if not spec.channel_id.strip():
        errors.append("channel_id must not be blank")
    if not spec.format_id.strip():
        errors.append("format_id must not be blank")

    try:
        channel = get_channel_preset(spec.channel_id)
        if channel.vendor != format_spec.vendor:
            errors.append(
                f"channel '{spec.channel_id}' expects vendor '{channel.vendor}' "
                f"but format '{spec.format_id}' is vendor '{format_spec.vendor}'"
            )
    except KeyError as exc:
        errors.append(str(exc))

    if spec.paper_type not in format_spec.paper_options and spec.paper_type not in format_spec.color_options:
        errors.append(
            f"paper_type '{spec.paper_type}' is not supported by format '{spec.format_id}'"
        )

    cfg = spec.layout_config

    if (cfg.rows is None) != (cfg.cols is None):
        errors.append("layout_config rows and cols must be provided together")

    if cfg.rows is not None and int(cfg.rows) <= 0:
        errors.append("layout_config.rows must be > 0")
    if cfg.cols is not None and int(cfg.cols) <= 0:
        errors.append("layout_config.cols must be > 0")
    if cfg.puzzles_per_page is not None and int(cfg.puzzles_per_page) <= 0:
        errors.append("layout_config.puzzles_per_page must be > 0")

    for field_name in [
        "inner_margin_in",
        "outer_margin_in",
        "top_margin_in",
        "bottom_margin_in",
        "header_height_in",
        "footer_height_in",
        "gutter_x_in",
        "gutter_y_in",
        "tile_slot_padding_in",
        "tile_header_band_height_in",
        "tile_gap_below_header_in",
        "tile_bottom_padding_in",
    ]:
        value = getattr(cfg, field_name)
        if value is not None and float(value) < 0:
            errors.append(f"layout_config.{field_name} must be >= 0")

    if cfg.given_digit_scale is not None and float(cfg.given_digit_scale) <= 0:
        errors.append("layout_config.given_digit_scale must be > 0")
    if cfg.solution_digit_scale is not None and float(cfg.solution_digit_scale) <= 0:
        errors.append("layout_config.solution_digit_scale must be > 0")

    if not is_supported_font_family(cfg.font_family):
        errors.append("layout_config.font_family must be one of: helvetica, times, courier")

    if not is_supported_digit_preset(cfg.given_digit_size_preset):
        errors.append("layout_config.given_digit_size_preset must be one of: small, medium, large, very_large")

    if not is_supported_digit_preset(cfg.solution_digit_size_preset):
        errors.append("layout_config.solution_digit_size_preset must be one of: small, medium, large, very_large")

    if (
        cfg.rows is not None
        and cfg.cols is not None
        and cfg.puzzles_per_page is not None
        and (int(cfg.rows) * int(cfg.cols)) != int(cfg.puzzles_per_page)
    ):
        errors.append("layout_config rows*cols must equal layout_config.puzzles_per_page")

    geometry = validate_layout_geometry(
        page_size=(format_spec.trim_width_in * 72.0, format_spec.trim_height_in * 72.0),
        layout_config=cfg,
        mirror_margins=bool(spec.mirror_margins),
    )
    if not geometry.ok:
        errors.append(f"layout_config geometry invalid: {geometry.message}")

    return errors