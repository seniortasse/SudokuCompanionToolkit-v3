from __future__ import annotations

import argparse
from pathlib import Path


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a publication package interior to a printable PDF."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    parser.add_argument(
        "--output-pdf",
        required=True,
        help="Path to the output PDF file.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    publication_dir = Path(args.publication_dir)
    output_pdf = Path(args.output_pdf)

    _log("=" * 72)
    _log("export_book_interior.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log(f"Output PDF:      {output_pdf.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    try:
        from python.publishing.pdf_renderer.compat import patch_hashlib_usedforsecurity
        patch_hashlib_usedforsecurity()
    except Exception as exc:
        _log(f"ERROR while installing hashlib compatibility shim: {exc}")
        return 1

    try:
        from python.publishing.pdf_renderer.interior_pdf_exporter import export_book_interior_pdf
    except ModuleNotFoundError as exc:
        if exc.name == "reportlab":
            _log("ERROR: Missing dependency 'reportlab'.")
            _log("Install it in your active virtual environment with:")
            _log("  python -m pip install reportlab")
            return 1
        raise

    try:
        written = export_book_interior_pdf(
            publication_dir=publication_dir,
            output_pdf_path=output_pdf,
        )
    except Exception as exc:
        _log(f"ERROR while exporting interior PDF: {exc}")
        return 1

    _log(f"Interior PDF written successfully: {written}")
    _log("=" * 72)
    _log("export_book_interior.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())