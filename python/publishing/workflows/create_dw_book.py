from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.book_builder.dw_presets import build_dw_preset_book_spec, list_dw_presets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a DW book spec from a preset template.")
    parser.add_argument("--library-short", default="CL9", help="Library short code, e.g. CL9")
    parser.add_argument("--aisle-short", default="DW", help="Aisle short code, e.g. DW")
    parser.add_argument("--book-number", type=int, required=True, help="Book number, e.g. 1")
    parser.add_argument(
        "--preset",
        required=True,
        help=f"DW preset name. Supported: {', '.join(list_dw_presets())}",
    )
    parser.add_argument("--puzzles-per-section", type=int, required=True, help="Puzzle count per section")
    parser.add_argument("--title", default=None, help="Optional book title override")
    parser.add_argument("--subtitle", default="", help="Optional book subtitle")
    parser.add_argument("--series-name", default="", help="Optional series name")
    parser.add_argument("--volume-number", type=int, default=None, help="Optional volume number")
    parser.add_argument("--description", default="", help="Optional description")
    parser.add_argument("--target-audience", default="general", help="Target audience")
    parser.add_argument("--trim-size", default="8.5x11", help="Trim size")
    parser.add_argument("--puzzles-per-page", type=int, default=1, help="Puzzles per page")
    parser.add_argument("--page-layout-profile", default="classic_single", help="Page layout profile")
    parser.add_argument("--solution-section-policy", default="appendix", help="Solution section policy")
    parser.add_argument("--cover-theme", default="classic", help="Cover theme")
    parser.add_argument("--layout-type", default="classic9x9", help="Layout type")
    parser.add_argument("--grid-size", type=int, default=9, help="Grid size")
    parser.add_argument(
        "--output-spec",
        required=True,
        help="Path where the generated book spec JSON should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    spec = build_dw_preset_book_spec(
        library_short=args.library_short,
        aisle_short=args.aisle_short,
        book_number=args.book_number,
        preset_name=args.preset,
        puzzles_per_section=args.puzzles_per_section,
        trim_size=args.trim_size,
        puzzles_per_page=args.puzzles_per_page,
        page_layout_profile=args.page_layout_profile,
        solution_section_policy=args.solution_section_policy,
        cover_theme=args.cover_theme,
        layout_type=args.layout_type,
        grid_size=args.grid_size,
        title=args.title,
        subtitle=args.subtitle,
        series_name=args.series_name,
        volume_number=args.volume_number,
        description=args.description,
        target_audience=args.target_audience,
        search_tags=["dw", args.preset],
    )

    output_path = Path(args.output_spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72, flush=True)
    print("create_dw_book.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Preset:      {args.preset}", flush=True)
    print(f"Book id:     {spec.book_id}", flush=True)
    print(f"Sections:    {len(spec.sections)}", flush=True)
    print(f"Output spec: {output_path}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())