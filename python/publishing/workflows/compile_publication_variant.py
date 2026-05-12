from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.publication_builder.layout_presets import list_layout_preset_ids
from python.publishing.publication_builder.spec_overrides import write_publication_spec_dict
from python.publishing.publication_builder.variant_compiler import compile_publication_variant_spec
from python.publishing.publication_builder.variant_models import PublicationVariantRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a resolved publication variant spec from a validated base publication spec."
    )
    parser.add_argument(
        "--base-spec",
        required=True,
        help="Path to the validated base publication spec JSON file.",
    )
    parser.add_argument(
        "--output-spec",
        required=True,
        help="Path where the resolved variant publication spec JSON should be written.",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Locale token used for identity and locale-pack discovery (examples: fr, de, it).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Human-readable language label for metadata.language.",
    )
    parser.add_argument(
        "--locale-pack",
        default=None,
        help="Optional explicit locale-pack JSON path. If omitted, the compiler auto-discovers it.",
    )
    parser.add_argument(
        "--layout",
        default=None,
        help=f"Layout preset id. Supported: {', '.join(list_layout_preset_ids())}",
    )
    parser.add_argument(
        "--puzzles-per-page",
        type=int,
        default=None,
        help="Optional direct puzzles-per-page override. Supported values: 1, 2, 4, 6, 12.",
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--font-family", default=None)
    parser.add_argument(
        "--publication-id",
        default=None,
        help="Optional explicit publication id. If omitted, a deterministic one is derived.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    base_spec_path = Path(args.base_spec).resolve()
    output_spec_path = Path(args.output_spec).resolve()
    locale_pack_path = Path(args.locale_pack).resolve() if args.locale_pack else None

    print("=" * 72)
    print("compile_publication_variant.py starting")
    print("=" * 72)
    print(f"Base spec:   {base_spec_path}")
    print(f"Output spec: {output_spec_path}")
    if locale_pack_path is not None:
        print(f"Locale pack: {locale_pack_path}")
    if args.layout:
        print(f"Layout:      {args.layout}")
    print("=" * 72)

    request = PublicationVariantRequest(
        base_spec_path=base_spec_path,
        locale=args.locale,
        language=args.language,
        locale_pack_path=locale_pack_path,
        layout_preset_id=args.layout,
        puzzles_per_page=args.puzzles_per_page,
        rows=args.rows,
        cols=args.cols,
        font_family=args.font_family,
        publication_id=args.publication_id,
    )

    resolved_spec = compile_publication_variant_spec(request)

    output_spec_path.parent.mkdir(parents=True, exist_ok=True)
    write_publication_spec_dict(resolved_spec, output_spec_path)

    print(f"Resolved publication_id: {resolved_spec['publication_id']}")
    print(f"Resolved language code:  {(resolved_spec.get('layout_config') or {}).get('language')}")
    print(f"Resolved language label: {(resolved_spec.get('metadata') or {}).get('language')}")
    print(f"Resolved puzzles/page:   {(resolved_spec.get('layout_config') or {}).get('puzzles_per_page')}")
    print(f"Resolved rows x cols:    {(resolved_spec.get('layout_config') or {}).get('rows')} x {(resolved_spec.get('layout_config') or {}).get('cols')}")
    print(f"Puzzle template:         {resolved_spec.get('puzzle_page_template')}")
    print(f"Solution template:       {resolved_spec.get('solution_page_template')}")
    print(f"Format id:               {resolved_spec.get('format_id')}")
    print(f"Paper type:              {resolved_spec.get('paper_type')}")
    print("=" * 72)
    print("compile_publication_variant.py completed successfully")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())