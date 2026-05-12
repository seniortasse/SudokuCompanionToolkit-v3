from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.cover_designs.cover_design_registry import (
    DEFAULT_COVER_DESIGN_CATALOG,
)
from python.publishing.cover_designs.cover_design_resolver import (
    resolved_context_to_dict,
)
from python.publishing.cover_designs.cover_design_validator import (
    validate_resolved_cover_design_context,
)
from python.publishing.cover_designs.cover_puzzle_art_report import (
    build_cover_puzzle_art_report,
)
from python.publishing.cover_designs.cover_puzzle_art_resolver import (
    resolve_cover_puzzle_art_variables,
)
from python.publishing.cover_designs.full_wrap_cover_renderer import (
    build_full_wrap_geometry,
    render_full_wrap_cover_pdf,
)

from python.publishing.cover_designs.publication_cover_design import (
    resolve_cover_design_context_from_publication_spec,
)
from python.publishing.cover_renderers.renderer_registry import get_cover_renderer


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))



def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _write_effective_publication_spec(
    *,
    publication_spec: dict,
    out_dir: Path,
    cover_design_id: str | None,
    cover_variables_json: str | None,
) -> Path | None:
    if not cover_design_id and not cover_variables_json:
        return None

    spec = dict(publication_spec)
    cover_design = dict(spec.get("cover_design") or {})

    if cover_design_id:
        cover_design["cover_design_id"] = cover_design_id

    if cover_variables_json:
        override_path = Path(cover_variables_json)
        override_variables = json.loads(override_path.read_text(encoding="utf-8"))
        existing_variables = dict(cover_design.get("variables") or {})
        cover_design["variables"] = _deep_merge(existing_variables, override_variables)

    spec["cover_design"] = cover_design

    effective_path = out_dir / "_effective_publication_spec.cover_override.json"
    effective_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return effective_path


def _count_pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return len(reader.pages)
    except ModuleNotFoundError:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return len(reader.pages)
    except ModuleNotFoundError:
        pass

    # Lightweight fallback: count PDF page objects.
    # This is good enough for generated ReportLab-style interiors.
    data = path.read_bytes()
    count = data.count(b"/Type /Page")
    pages_count = data.count(b"/Type /Pages")

    page_count = count - pages_count
    if page_count > 0:
        return page_count

    raise RuntimeError(
        "Could not count pages from interior PDF. Install pypdf/PyPDF2 "
        "or rerun with --page-count."
    )


def _publication_text(publication_spec: dict) -> dict:
    metadata = dict(publication_spec.get("metadata") or {})
    editorial = dict(publication_spec.get("editorial_copy") or {})
    brand = dict(editorial.get("brand") or {})

    title = (
        metadata.get("title")
        or publication_spec.get("title")
        or publication_spec.get("book_id")
        or ""
    )
    spine_text = (
        metadata.get("spine_text")
        or brand.get("spine_text")
        or title
        or ""
    )
    back_copy = (
        metadata.get("back_copy")
        or publication_spec.get("back_copy")
        or ""
    )

    return {
        "title": str(title),
        "spine_text": str(spine_text),
        "back_copy": str(back_copy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a full-wrap cover PDF using publication_spec.cover_design."
    )
    parser.add_argument(
        "--publication-spec",
        required=True,
        help="Path to publication spec JSON containing cover_design block.",
    )
    parser.add_argument(
        "--book-dir",
        default=None,
        help="Optional built book directory used to resolve puzzle-driven cover grid art.",
    )
    parser.add_argument(
        "--interior-pdf",
        default=None,
        help="Interior PDF used to compute page count/spine width.",
    )
    parser.add_argument(
        "--page-count",
        type=int,
        default=None,
        help="Manual page count. Used only when --interior-pdf is not supplied.",
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_COVER_DESIGN_CATALOG),
        help="Path to cover_design_catalog.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for full-wrap cover assets.",
    )
    parser.add_argument(
        "--puzzle-records-dir",
        default=None,
        help="Optional puzzle_records directory.",
    )
    parser.add_argument("--trim-width-in", type=float, default=8.5)
    parser.add_argument("--trim-height-in", type=float, default=11.0)
    parser.add_argument("--bleed-in", type=float, default=0.125)
    parser.add_argument("--front-width-px", type=int, default=2550)
    parser.add_argument("--front-height-px", type=int, default=3300)

    parser.add_argument(
        "--cover-design-id",
        default=None,
        help="Optional cover design override for previewing another catalog cover without editing the publication spec.",
    )
    parser.add_argument(
        "--cover-variables-json",
        default=None,
        help="Optional JSON file containing cover_design.variables overrides.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    publication_spec_path = Path(args.publication_spec)
    publication_spec = _load_json(publication_spec_path)

    effective_publication_spec_path = _write_effective_publication_spec(
        publication_spec=publication_spec,
        out_dir=out_dir,
        cover_design_id=args.cover_design_id,
        cover_variables_json=args.cover_variables_json,
    )

    context_publication_spec_path = effective_publication_spec_path or publication_spec_path

    if args.interior_pdf:
        page_count = _count_pdf_pages(Path(args.interior_pdf))
    elif args.page_count is not None:
        page_count = args.page_count
    else:
        raise SystemExit("Provide either --interior-pdf or --page-count so spine width can be computed.")

    context = resolve_cover_design_context_from_publication_spec(
        context_publication_spec_path,
        catalog_path=args.catalog,
    )

    if context is None:
        raise SystemExit(
            "No publication_spec.cover_design block found. "
            "Cannot generate full-wrap cover with the cover design pipeline."
        )

    errors = validate_resolved_cover_design_context(context)
    if errors:
        print("Validation: FAILED")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    if args.book_dir:
        context.variables["resolved_puzzle_art"] = resolve_cover_puzzle_art_variables(
            context_variables=context.variables,
            book_dir=Path(args.book_dir),
            puzzle_records_dir=Path(args.puzzle_records_dir) if args.puzzle_records_dir else None,
        )

    puzzle_art_report = build_cover_puzzle_art_report(
        context.variables.get("resolved_puzzle_art")
    )



    front_art_dir = out_dir / "front_art"
    front_art_dir.mkdir(parents=True, exist_ok=True)

    renderer = get_cover_renderer(context.renderer_key)
    front_result = renderer.render_front_cover(
        context=context,
        out_dir=front_art_dir,
        width_px=args.front_width_px,
        height_px=args.front_height_px,
    )

    wrap_geometry = build_full_wrap_geometry(
        page_count=page_count,
        paper_type=str(publication_spec.get("paper_type", "white_bw")),
        trim_width_in=args.trim_width_in,
        trim_height_in=args.trim_height_in,
        bleed_in=args.bleed_in,
    )

    panel_art_dir = out_dir / "panel_art"
    panel_art_dir.mkdir(parents=True, exist_ok=True)

    back_cover_png = renderer.render_back_cover(
        context=context,
        out_dir=panel_art_dir,
        geometry=wrap_geometry,
        width_px=args.front_width_px,
        height_px=args.front_height_px,
    )
    if back_cover_png is None:
        back_cover_png = front_result.back_cover_png

    spine_width_px = max(
        1,
        int(round(args.front_width_px * wrap_geometry.spine_width_in / args.trim_width_in)),
    )
    spine_cover_png = renderer.render_spine_cover(
        context=context,
        out_dir=panel_art_dir,
        geometry=wrap_geometry,
        width_px=spine_width_px,
        height_px=args.front_height_px,
    )
    if spine_cover_png is None:
        spine_cover_png = front_result.spine_cover_png

    context_path = out_dir / "cover_design_context.json"
    context_path.write_text(
        json.dumps(resolved_context_to_dict(context), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    puzzle_report_path = out_dir / "cover_puzzle_art_report.json"
    puzzle_report_path.write_text(
        json.dumps(puzzle_art_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pub_text = _publication_text(publication_spec)

    output_pdf = out_dir / "cover.pdf"
    geometry_metadata = render_full_wrap_cover_pdf(
        front_cover_png=front_result.front_cover_png,
        output_pdf=output_pdf,
        page_count=page_count,
        paper_type=str(publication_spec.get("paper_type", "white_bw")),
        title=pub_text["title"],
        spine_text=pub_text["spine_text"],
        back_copy=pub_text["back_copy"],
        trim_width_in=args.trim_width_in,
        trim_height_in=args.trim_height_in,
        bleed_in=args.bleed_in,
        back_cover_png=back_cover_png,
        spine_cover_png=spine_cover_png,
    )

    validation_report_path = out_dir / "cover_validation_report.json"

    manifest_path = out_dir / "generated_full_wrap_cover_assets.json"
    manifest = {
        "cover_pdf": str(output_pdf),
        "front_cover_png": str(front_result.front_cover_png),
        "back_cover_png": str(back_cover_png) if back_cover_png is not None else None,
        "spine_cover_png": str(spine_cover_png) if spine_cover_png is not None else None,
        "cover_design_context_json": str(context_path),
        "cover_puzzle_art_report_json": str(puzzle_report_path),
        "geometry_json": str(output_pdf.with_suffix(".geometry.json")),
        "validation_report_json": str(validation_report_path),
        "page_count": page_count,
        "paper_type": str(publication_spec.get("paper_type", "white_bw")),
        "geometry": geometry_metadata["geometry"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("Generated full-wrap cover PDF")
    print("=" * 72)
    print(f"cover_design_id: {context.cover_design_id}")
    print(f"renderer_key:    {context.renderer_key}")
    print(f"page_count:      {page_count}")
    print(f"cover_pdf:       {output_pdf}")
    print(f"front_cover_png: {front_result.front_cover_png}")
    print(f"back_cover_png:  {back_cover_png}")
    print(f"spine_cover_png: {spine_cover_png}")
    print(f"context_json:    {context_path}")
    print(f"puzzle_report:   {puzzle_report_path}")
    print(f"manifest_json:   {manifest_path}")
    print(f"puzzle_status:   {puzzle_art_report.get('status')}")


if __name__ == "__main__":
    main()