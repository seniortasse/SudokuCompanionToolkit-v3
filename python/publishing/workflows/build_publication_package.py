from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from python.publishing.publication_builder.publication_package_builder import build_publication_package
from python.publishing.publication_builder.spec_overrides import (
    apply_publication_spec_overrides,
    load_publication_spec_dict,
    write_publication_spec_dict,
)


def _parse_bool_flag(value: str) -> bool:
    key = str(value).strip().lower()
    if key in {"true", "1", "yes", "y"}:
        return True
    if key in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a publication package from a built book directory and a publication spec."
    )
    parser.add_argument("--book-dir", required=True, help="Path to the built book directory.")
    parser.add_argument("--publication-spec", required=True, help="Path to the publication spec JSON file.")
    parser.add_argument("--output-publications-dir", required=True, help="Directory where publication packages are written.")

    parser.add_argument("--include-solutions", type=_parse_bool_flag, default=None)
    parser.add_argument("--page-numbering-policy", default=None)

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
    publication_spec_path = Path(args.publication_spec).resolve()
    output_publications_dir = Path(args.output_publications_dir).resolve()

    print("=" * 72)
    print("build_publication_package.py starting")
    print("=" * 72)
    print(f"Book dir:                {book_dir}")
    print(f"Publication spec:        {publication_spec_path}")
    print(f"Output publications dir: {output_publications_dir}")
    print("=" * 72)

    base_spec = load_publication_spec_dict(publication_spec_path)
    merged_spec = apply_publication_spec_overrides(
        base_spec,
        include_solutions=args.include_solutions,
        page_numbering_policy=args.page_numbering_policy,
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

    with tempfile.TemporaryDirectory(prefix="publication_spec_override_") as tmpdir:
        temp_spec_path = Path(tmpdir) / publication_spec_path.name
        write_publication_spec_dict(merged_spec, temp_spec_path)

        publication_dir, _package = build_publication_package(
            book_dir=book_dir,
            publication_spec_path=temp_spec_path,
            output_publications_dir=output_publications_dir,
        )

    print(f"Publication dir:         {publication_dir}")
    print("=" * 72)
    print("build_publication_package.py completed successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()