from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.cover_designs.cover_design_registry import (
    DEFAULT_COVER_DESIGN_CATALOG,
    load_cover_design_catalog,
    load_cover_design_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List cover designs registered in the cover design catalog."
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_COVER_DESIGN_CATALOG),
        help="Path to cover_design_catalog.json.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print editable variable groups for each design.",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    entries = load_cover_design_catalog(catalog_path)

    print("=" * 72)
    print("Cover design catalog")
    print("=" * 72)
    print(f"Catalog: {catalog_path}")
    print(f"Designs: {len(entries)}")
    print()

    for entry in entries:
        print(f"- {entry.cover_design_id}")
        print(f"  name:         {entry.name}")
        print(f"  family:       {entry.family}")
        print(f"  renderer_key: {entry.renderer_key}")
        print(f"  status:       {entry.status}")

        if args.details:
            record = load_cover_design_record(entry.cover_design_id, catalog_path)
            variable_names = sorted(record.editable_variables.keys())
            signature = record.identity.get("visual_signature", [])

            print(f"  variables:    {', '.join(variable_names)}")

            if signature:
                print("  signature:")
                for item in signature:
                    print(f"    - {item}")

        print()


if __name__ == "__main__":
    main()