from __future__ import annotations

import argparse
from pathlib import Path


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export structured distribution metadata for a publication package."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where metadata exports should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publication_dir = Path(args.publication_dir)
    output_dir = Path(args.output_dir)

    _log("=" * 72)
    _log("export_distribution_metadata.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log(f"Output dir:      {output_dir.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    try:
        from python.publishing.distribution import export_publication_metadata
    except Exception as exc:
        _log(f"ERROR importing distribution exporter: {exc}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        metadata_path = export_publication_metadata(
            publication_dir=publication_dir,
            output_path=output_dir / "metadata.json",
        )
    except Exception as exc:
        _log(f"ERROR while exporting metadata: {exc}")
        return 1

    _log(f"Metadata JSON:   {metadata_path}")
    kdp_profile = output_dir / "kdp_profile.json"
    if kdp_profile.exists():
        _log(f"KDP profile:     {kdp_profile}")

    _log("=" * 72)
    _log("export_distribution_metadata.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())