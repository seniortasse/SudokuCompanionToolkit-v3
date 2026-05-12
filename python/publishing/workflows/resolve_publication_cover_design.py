from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.cover_designs.cover_design_registry import (
    DEFAULT_COVER_DESIGN_CATALOG,
)
from python.publishing.cover_designs.cover_design_resolver import resolved_context_to_dict
from python.publishing.cover_designs.cover_design_validator import (
    validate_resolved_cover_design_context,
)
from python.publishing.cover_designs.publication_cover_design import (
    load_publication_spec,
    resolve_cover_design_context_from_publication_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve cover_design block from a publication spec."
    )
    parser.add_argument(
        "--publication-spec",
        required=True,
        help="Path to publication spec JSON.",
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_COVER_DESIGN_CATALOG),
        help="Path to cover_design_catalog.json.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output path for resolved cover design context JSON.",
    )
    args = parser.parse_args()

    publication_spec_path = Path(args.publication_spec)
    publication_spec = load_publication_spec(publication_spec_path)

    print("=" * 72)
    print("Publication cover design resolution")
    print("=" * 72)
    print(f"Publication spec: {publication_spec_path}")
    print(f"publication_id:   {publication_spec.get('publication_id')}")
    print(f"book_id:          {publication_spec.get('book_id')}")
    print(f"cover_template:   {publication_spec.get('cover_template')}")
    print()

    context = resolve_cover_design_context_from_publication_spec(
        publication_spec_path,
        catalog_path=args.catalog,
    )

    if context is None:
        print("No cover_design block found.")
        print("Legacy cover behavior remains active.")
        return

    errors = validate_resolved_cover_design_context(context)

    print(f"cover_design_id: {context.cover_design_id}")
    print(f"name:            {context.name}")
    print(f"family:          {context.family}")
    print(f"renderer_key:    {context.renderer_key}")
    print(f"status:          {context.status}")
    print()

    text = context.variables.get("text", {})
    print("Resolved text:")
    print(f"  year:               {text.get('year')}")
    print(f"  puzzle_count_label: {text.get('puzzle_count_label')}")
    print(f"  title_word:         {text.get('title_word')}")
    print(f"  title_joiner:       {text.get('title_joiner')!r}")
    print(f"  difficulty_label:   {text.get('difficulty_label')}")
    print()

    print(f"palette_id: {context.variables.get('palette_id')}")
    print(f"texture_id: {context.variables.get('texture_id')}")
    print()

    if errors:
        print("Validation: FAILED")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Validation: OK")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(resolved_context_to_dict(context), f, indent=2, ensure_ascii=False)
        print()
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()