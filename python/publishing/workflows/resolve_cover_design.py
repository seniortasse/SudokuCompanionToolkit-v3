from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.publishing.cover_designs.cover_design_registry import (
    DEFAULT_COVER_DESIGN_CATALOG,
)
from python.publishing.cover_designs.cover_design_resolver import (
    load_cover_design_assignment,
    resolve_cover_design_context,
    resolved_context_to_dict,
)
from python.publishing.cover_designs.cover_design_validator import (
    validate_resolved_cover_design_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a cover design assignment into renderer-ready context."
    )
    parser.add_argument(
        "--assignment",
        required=True,
        help="Path to a cover design assignment JSON file.",
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_COVER_DESIGN_CATALOG),
        help="Path to cover_design_catalog.json.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path where resolved cover design context JSON should be written.",
    )
    args = parser.parse_args()

    assignment = load_cover_design_assignment(args.assignment)
    context = resolve_cover_design_context(assignment, args.catalog)
    errors = validate_resolved_cover_design_context(context)
    payload = resolved_context_to_dict(context)

    print("=" * 72)
    print("Resolved cover design context")
    print("=" * 72)
    print(f"cover_design_id: {context.cover_design_id}")
    print(f"name:            {context.name}")
    print(f"family:          {context.family}")
    print(f"renderer_key:    {context.renderer_key}")
    print(f"status:          {context.status}")
    print(f"assignment:      {context.assignment_source}")
    print()

    text = context.variables.get("text", {})
    print("Text variables:")
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
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print()
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()