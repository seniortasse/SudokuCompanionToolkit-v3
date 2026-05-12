from __future__ import annotations

import argparse
from pathlib import Path


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export lightweight preview PNGs for a publication package."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where preview PNGs should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publication_dir = Path(args.publication_dir)
    output_dir = Path(args.output_dir)

    _log("=" * 72)
    _log("export_publication_previews.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log(f"Output dir:      {output_dir.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    try:
        from python.publishing.distribution import export_publication_previews
    except ModuleNotFoundError as exc:
        _log(f"ERROR importing preview exporter: {exc}")
        return 1

    try:
        results = export_publication_previews(
            publication_dir=publication_dir,
            output_dir=output_dir,
        )
    except Exception as exc:
        _log(f"ERROR while exporting previews: {exc}")
        return 1

    for key, value in results.items():
        _log(f"{key}: {value}")

    _log("=" * 72)
    _log("export_publication_previews.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())