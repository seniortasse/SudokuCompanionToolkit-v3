from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.pdf_renderer.book_pdf_exporter import export_book_pdf


def _parse_bool_flag(value: str) -> bool:
    key = str(value).strip().lower()
    if key in {"true", "1", "yes", "y"}:
        return True
    if key in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a built book package to a printable PDF via the legacy-compatible wrapper."
    )
    parser.add_argument("--book-dir", required=True, help="Path to the built book directory.")
    parser.add_argument("--output-pdf", required=True, help="Path to the output PDF file.")

    parser.add_argument("--page-numbering-policy", default="physical_all_suppress_blank_only")
    parser.add_argument("--include-solutions", type=_parse_bool_flag, default=None)

    parser.add_argument("--puzzles-per-page", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)

    parser.add_argument("--gutter-x-in", type=float, default=None)
    parser.add_argument("--gutter-y-in", type=float, default=None)

    parser.add_argument("--inner-margin-in", type=float, default=None)
    parser.add_argument("--outer-margin-in", type=float, default=None)
    parser.add_argument("--top-margin-in", type=float, default=None)
    parser.add_argument("--bottom-margin-in", type=float, default=None)

    parser.add_argument("--header-height-in", type=float, default=None)
    parser.add_argument("--footer-height-in", type=float, default=None)

    parser.add_argument("--tile-slot-padding-in", type=float, default=None)
    parser.add_argument("--tile-header-band-height-in", type=float, default=None)
    parser.add_argument("--tile-gap-below-header-in", type=float, default=None)
    parser.add_argument("--tile-bottom-padding-in", type=float, default=None)

    parser.add_argument("--font-family", default=None)
    parser.add_argument("--language", default=None)

    parser.add_argument("--given-digit-size-preset", default=None)
    parser.add_argument("--solution-digit-size-preset", default=None)
    parser.add_argument("--given-digit-scale", type=float, default=None)
    parser.add_argument("--solution-digit-scale", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    book_dir = Path(args.book_dir).resolve()
    output_pdf = Path(args.output_pdf).resolve()

    print("=" * 72)
    print("export_book_pdf.py starting")
    print("=" * 72)
    print(f"Book dir:   {book_dir}")
    print(f"Output PDF: {output_pdf}")
    print("=" * 72)

    written = export_book_pdf(
        book_dir=book_dir,
        output_pdf_path=output_pdf,
        page_numbering_policy=args.page_numbering_policy,
        include_solutions=args.include_solutions,
        puzzles_per_page=args.puzzles_per_page,
        rows=args.rows,
        cols=args.cols,
        gutter_x_in=args.gutter_x_in,
        gutter_y_in=args.gutter_y_in,
        inner_margin_in=args.inner_margin_in,
        outer_margin_in=args.outer_margin_in,
        top_margin_in=args.top_margin_in,
        bottom_margin_in=args.bottom_margin_in,
        header_height_in=args.header_height_in,
        footer_height_in=args.footer_height_in,
        tile_slot_padding_in=args.tile_slot_padding_in,
        tile_header_band_height_in=args.tile_header_band_height_in,
        tile_gap_below_header_in=args.tile_gap_below_header_in,
        tile_bottom_padding_in=args.tile_bottom_padding_in,
        font_family=args.font_family,
        language=args.language,
        given_digit_size_preset=args.given_digit_size_preset,
        solution_digit_size_preset=args.solution_digit_size_preset,
        given_digit_scale=args.given_digit_scale,
        solution_digit_scale=args.solution_digit_scale,
    )

    print(f"PDF written successfully: {written}")
    print("=" * 72)
    print("export_book_pdf.py completed successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()