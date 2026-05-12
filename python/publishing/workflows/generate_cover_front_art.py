from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.cover_designs.cover_puzzle_art_report import (
    build_cover_puzzle_art_report,
)
from python.publishing.cover_designs.cover_puzzle_art_resolver import resolve_cover_puzzle_art_variables

from python.publishing.cover_designs.cover_design_registry import (
    DEFAULT_COVER_DESIGN_CATALOG,
)
from python.publishing.cover_designs.cover_design_resolver import (
    resolved_context_to_dict,
)
from python.publishing.cover_designs.cover_design_validator import (
    validate_resolved_cover_design_context,
)
from python.publishing.cover_designs.publication_cover_design import (
    resolve_cover_design_context_from_publication_spec,
)
from python.publishing.cover_renderers.renderer_registry import get_cover_renderer


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
    publication_spec_path: Path,
    out_dir: Path,
    cover_design_id: str | None,
    cover_variables_json: str | None,
) -> Path:
    if not cover_design_id and not cover_variables_json:
        return publication_spec_path

    spec = json.loads(publication_spec_path.read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate front-cover artwork from publication_spec.cover_design."
    )
    parser.add_argument(
        "--publication-spec",
        required=True,
        help="Path to publication spec JSON containing cover_design block.",
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_COVER_DESIGN_CATALOG),
        help="Path to cover_design_catalog.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for generated cover assets.",
    )
    parser.add_argument(
        "--width-px",
        type=int,
        default=2550,
        help="Front cover width in pixels. Default is 8.5in at 300 DPI.",
    )
    parser.add_argument(
        "--height-px",
        type=int,
        default=3300,
        help="Front cover height in pixels. Default is 11in at 300 DPI.",
    )

    parser.add_argument(
        "--book-dir",
        default=None,
        help="Optional built book directory used to resolve puzzle-driven cover grid art.",
    )
    parser.add_argument(
        "--puzzle-records-dir",
        default=None,
        help="Optional puzzle_records directory. Defaults to sibling puzzle_records next to books/.",
    )

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
    effective_publication_spec_path = _write_effective_publication_spec(
        publication_spec_path=publication_spec_path,
        out_dir=out_dir,
        cover_design_id=args.cover_design_id,
        cover_variables_json=args.cover_variables_json,
    )

    context = resolve_cover_design_context_from_publication_spec(
        effective_publication_spec_path,
        catalog_path=args.catalog,
    )

    if context is None:
        raise SystemExit(
            "No publication_spec.cover_design block found. "
            "Nothing to render with the new cover design pipeline."
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

    renderer = get_cover_renderer(context.renderer_key)

    result = renderer.render_front_cover(
        context=context,
        out_dir=out_dir,
        width_px=args.width_px,
        height_px=args.height_px,
    )

    context_path = out_dir / "cover_design_context.json"
    with context_path.open("w", encoding="utf-8") as f:
        json.dump(resolved_context_to_dict(context), f, indent=2, ensure_ascii=False)

    puzzle_art_report_path = out_dir / "cover_puzzle_art_report.json"
    with puzzle_art_report_path.open("w", encoding="utf-8") as f:
        json.dump(puzzle_art_report, f, indent=2, ensure_ascii=False)

    manifest_path = out_dir / "generated_cover_assets.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "renderer_key": result.renderer_key,
                "front_cover_png": str(result.front_cover_png),
                "width_px": result.width_px,
                "height_px": result.height_px,
                "context_json": str(context_path),
                "cover_puzzle_art_report_json": str(puzzle_art_report_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("Generated front-cover artwork")
    print("=" * 72)
    print(f"cover_design_id: {context.cover_design_id}")
    print(f"renderer_key:    {context.renderer_key}")
    print(f"front_cover_png: {result.front_cover_png}")
    print(f"context_json:    {context_path}")
    print(f"puzzle_report:   {puzzle_art_report_path}")
    print(f"manifest_json:   {manifest_path}")
    print(f"puzzle_status:   {puzzle_art_report.get('status')}")


if __name__ == "__main__":
    main()