from __future__ import annotations

import json
import tempfile
from pathlib import Path

from python.publishing.pdf_renderer.interior_pdf_exporter import export_book_interior_pdf
from python.publishing.pdf_renderer.render_models import load_built_book_render_model
from python.publishing.publication_builder.layout_presets import resolve_layout_preset
from python.publishing.publication_builder.publication_package_builder import build_publication_package


def export_book_pdf(
    *,
    book_dir: Path,
    output_pdf_path: Path,
    page_numbering_policy: str = "physical_all_suppress_blank_only",
    include_solutions: bool | None = None,
    puzzles_per_page: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    gutter_x_in: float | None = None,
    gutter_y_in: float | None = None,
    inner_margin_in: float | None = None,
    outer_margin_in: float | None = None,
    top_margin_in: float | None = None,
    bottom_margin_in: float | None = None,
    header_height_in: float | None = None,
    footer_height_in: float | None = None,
    tile_slot_padding_in: float | None = None,
    tile_header_band_height_in: float | None = None,
    tile_gap_below_header_in: float | None = None,
    tile_bottom_padding_in: float | None = None,
    font_family: str | None = None,
    language: str | None = None,
    given_digit_size_preset: str | None = None,
    solution_digit_size_preset: str | None = None,
    given_digit_scale: float | None = None,
    solution_digit_scale: float | None = None,
) -> Path:
    """
    Legacy-compatible wrapper.

    This function preserves the old book-dir based export flow, but now
    routes through a temporary publication package and the publication-first
    interior renderer introduced in Wave 2.
    """
    render_model = load_built_book_render_model(book_dir)
    book_manifest = render_model.book_manifest

    with tempfile.TemporaryDirectory(prefix="sudoku_publication_export_") as tmpdir:
        tmp_root = Path(tmpdir)
        publication_spec_path = tmp_root / "legacy_publication_spec.json"
        output_publications_dir = tmp_root / "publications"

        publication_spec = _build_legacy_publication_spec(
            render_model,
            page_numbering_policy=page_numbering_policy,
            include_solutions=include_solutions,
            puzzles_per_page=puzzles_per_page,
            rows=rows,
            cols=cols,
            gutter_x_in=gutter_x_in,
            gutter_y_in=gutter_y_in,
            inner_margin_in=inner_margin_in,
            outer_margin_in=outer_margin_in,
            top_margin_in=top_margin_in,
            bottom_margin_in=bottom_margin_in,
            header_height_in=header_height_in,
            footer_height_in=footer_height_in,
            tile_slot_padding_in=tile_slot_padding_in,
            tile_header_band_height_in=tile_header_band_height_in,
            tile_gap_below_header_in=tile_gap_below_header_in,
            tile_bottom_padding_in=tile_bottom_padding_in,
            font_family=font_family,
            language=language,
            given_digit_size_preset=given_digit_size_preset,
            solution_digit_size_preset=solution_digit_size_preset,
            given_digit_scale=given_digit_scale,
            solution_digit_scale=solution_digit_scale,
        )



        publication_spec_path.write_text(
            json.dumps(publication_spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        publication_dir, _package = build_publication_package(
            book_dir=book_dir,
            publication_spec_path=publication_spec_path,
            output_publications_dir=output_publications_dir,
        )

        written = export_book_interior_pdf(
            publication_dir=publication_dir,
            output_pdf_path=output_pdf_path,
        )

    return written


def _build_legacy_publication_spec(
    render_model,
    *,
    page_numbering_policy: str,
    include_solutions: bool | None = None,
    puzzles_per_page: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    gutter_x_in: float | None = None,
    gutter_y_in: float | None = None,
    inner_margin_in: float | None = None,
    outer_margin_in: float | None = None,
    top_margin_in: float | None = None,
    bottom_margin_in: float | None = None,
    header_height_in: float | None = None,
    footer_height_in: float | None = None,
    tile_slot_padding_in: float | None = None,
    tile_header_band_height_in: float | None = None,
    tile_gap_below_header_in: float | None = None,
    tile_bottom_padding_in: float | None = None,
    font_family: str | None = None,
    language: str | None = None,
    given_digit_size_preset: str | None = None,
    solution_digit_size_preset: str | None = None,
    given_digit_scale: float | None = None,
    solution_digit_scale: float | None = None,
) -> dict:
    book = render_model.book_manifest

    effective_puzzles_per_page = int(puzzles_per_page or book.puzzles_per_page)
    effective_include_solutions = (
        bool(include_solutions)
        if include_solutions is not None
        else (str(book.solution_section_policy).lower() != "none")
    )

    preset = resolve_layout_preset(puzzles_per_page=effective_puzzles_per_page)
    if preset is None:
        raise ValueError(
            f"Unsupported legacy export puzzles_per_page={effective_puzzles_per_page}. "
            "Supported values: 1, 2, 4, 6, 12."
        )

    resolved_rows = int(rows) if rows is not None else int(preset.rows)
    resolved_cols = int(cols) if cols is not None else int(preset.cols)

    return {
        "publication_id": f"LEGACY-{book.book_id}-INTERIOR",
        "book_id": book.book_id,
        "channel_id": "amazon_kdp_paperback",
        "format_id": _infer_format_id(book.trim_size),
        "include_cover": False,
        "include_solutions": effective_include_solutions,
        "mirror_margins": True,
        "front_matter_profile": "minimal_front_matter",
        "end_matter_profile": "none",
        "section_separator_policy": "section_openers",
        "blank_page_policy": "none",
        "page_numbering_policy": str(page_numbering_policy or "physical_all_suppress_blank_only"),
        "puzzle_page_template": preset.puzzle_page_template,
        "solution_page_template": preset.solution_page_template,
        "cover_template": "basic_full_wrap",
        "paper_type": "white_bw",
        "layout_config": {
            "puzzles_per_page": effective_puzzles_per_page,
            "rows": resolved_rows,
            "cols": resolved_cols,
            "inner_margin_in": preset.inner_margin_in if inner_margin_in is None else float(inner_margin_in),
            "outer_margin_in": preset.outer_margin_in if outer_margin_in is None else float(outer_margin_in),
            "top_margin_in": preset.top_margin_in if top_margin_in is None else float(top_margin_in),
            "bottom_margin_in": preset.bottom_margin_in if bottom_margin_in is None else float(bottom_margin_in),
            "header_height_in": preset.header_height_in if header_height_in is None else float(header_height_in),
            "footer_height_in": preset.footer_height_in if footer_height_in is None else float(footer_height_in),
            "gutter_x_in": preset.gutter_x_in if gutter_x_in is None else float(gutter_x_in),
            "gutter_y_in": preset.gutter_y_in if gutter_y_in is None else float(gutter_y_in),
            "tile_slot_padding_in": tile_slot_padding_in,
            "tile_header_band_height_in": tile_header_band_height_in,
            "tile_gap_below_header_in": tile_gap_below_header_in,
            "tile_bottom_padding_in": tile_bottom_padding_in,
            "font_family": font_family,
            "language": language,
            "given_digit_size_preset": given_digit_size_preset,
            "solution_digit_size_preset": solution_digit_size_preset,
            "given_digit_scale": given_digit_scale,
            "solution_digit_scale": solution_digit_scale,
        },
        "metadata": {
            "imprint_name": "Sudoku Companion",
        },
    }


def _infer_format_id(trim_size: str) -> str:
    key = str(trim_size).strip().lower().replace('"', "")
    if key in {"8.5x11", "8.5 x 11"}:
        return "amazon_kdp_paperback_8_5x11_bw"
    if key in {"6x9", "6 x 9"}:
        return "amazon_kdp_paperback_6x9_bw"
    return "amazon_kdp_paperback_8_5x11_bw"


def _infer_puzzle_template(page_layout_profile: str, puzzles_per_page: int) -> str:
    key = str(page_layout_profile).strip().lower()
    count = int(puzzles_per_page)

    if key in {"classic_two_up", "classic_2up_blackband"} or count == 2:
        return "classic_2up_blackband"
    if key in {"classic_four_up", "classic_4up_blackband"} or count == 4:
        return "classic_4up_blackband"
    if key in {"classic_six_up_blackband", "classic_6up_blackband"} or count == 6:
        return "classic_6up_blackband"
    if key in {"classic_twelve_up_blackband", "classic_12up_blackband"} or count == 12:
        return "classic_12up_blackband"

    return "classic_4up_blackband"


def _infer_solution_template(page_layout_profile: str, puzzles_per_page: int) -> str:
    key = str(page_layout_profile).strip().lower()
    count = int(puzzles_per_page)

    if key in {"classic_two_up", "solution_2up_blackband", "classic_2up_blackband"} or count == 2:
        return "solution_2up_blackband"
    if key in {"classic_four_up", "solution_4up_blackband", "classic_4up_blackband"} or count == 4:
        return "solution_4up_blackband"
    if key in {"classic_six_up_blackband", "solution_6up_blackband", "classic_6up_blackband"} or count == 6:
        return "solution_6up_blackband"
    if key in {"classic_twelve_up_blackband", "solution_12up_blackband", "classic_12up_blackband"} or count == 12:
        return "solution_12up_blackband"

    return "solution_4up_blackband"