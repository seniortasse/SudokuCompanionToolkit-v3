# tools/extract_cover_panels.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PT_PER_INCH = 72.0


def _require_pymupdf():
    try:
        import fitz  # PyMuPDF
        return fitz
    except ImportError:
        print(
            "ERROR: This script requires PyMuPDF.\n\n"
            "Install it with:\n"
            "  pip install pymupdf\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _inches_to_points(value: float) -> float:
    return value * PT_PER_INCH


def _safe_output_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 2
    while True:
        candidate = parent / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def extract_front_back_from_pdf(
    pdf_path: Path,
    *,
    output_dir: Path | None,
    dpi: int,
    trim_width_in: float,
    trim_height_in: float,
    bleed_in: float,
    front_side: str,
    overwrite: bool,
    dry_run: bool,
) -> list[Path]:
    fitz = _require_pymupdf()

    pdf_path = pdf_path.resolve()
    target_dir = output_dir.resolve() if output_dir else pdf_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    trim_w = _inches_to_points(trim_width_in)
    trim_h = _inches_to_points(trim_height_in)
    bleed = _inches_to_points(bleed_in)

    doc = fitz.open(pdf_path)

    if doc.page_count < 1:
        raise ValueError(f"{pdf_path} has no pages.")

    page = doc[0]
    page_rect = page.rect
    page_w = float(page_rect.width)
    page_h = float(page_rect.height)

    spine_w = page_w - (2.0 * trim_w) - (2.0 * bleed)

    if spine_w < -1.0:
        raise ValueError(
            f"{pdf_path.name}: PDF is too narrow for "
            f"2 x {trim_width_in}\" panels plus {bleed_in}\" bleed on both sides. "
            f"Page width is {page_w / PT_PER_INCH:.3f}\"."
        )

    # Small negative values can happen from rounding.
    spine_w = max(0.0, spine_w)

    if page_h + 1.0 < trim_h + (2.0 * bleed):
        raise ValueError(
            f"{pdf_path.name}: PDF is too short for "
            f"{trim_height_in}\" trim height plus {bleed_in}\" top/bottom bleed. "
            f"Page height is {page_h / PT_PER_INCH:.3f}\"."
        )

    # PyMuPDF page coordinates use top-left origin.
    left_panel = fitz.Rect(
        bleed,
        bleed,
        bleed + trim_w,
        bleed + trim_h,
    )

    right_panel = fitz.Rect(
        bleed + trim_w + spine_w,
        bleed,
        bleed + trim_w + spine_w + trim_w,
        bleed + trim_h,
    )

    if front_side == "right":
        clips = {
            "back": left_panel,
            "front": right_panel,
        }
    else:
        clips = {
            "front": left_panel,
            "back": right_panel,
        }

    zoom = dpi / PT_PER_INCH
    matrix = fitz.Matrix(zoom, zoom)

    created: list[Path] = []

    print(f"\nPDF: {pdf_path.name}")
    print(f"  Page size: {page_w / PT_PER_INCH:.3f}\" x {page_h / PT_PER_INCH:.3f}\"")
    print(f"  Trim panel: {trim_width_in}\" x {trim_height_in}\"")
    print(f"  Bleed removed: {bleed_in}\"")
    print(f"  Detected spine width: {spine_w / PT_PER_INCH:.3f}\"")
    print(f"  Front side: {front_side}")

    for label, clip in clips.items():
        out_path = target_dir / f"{pdf_path.stem}-{label}.png"
        out_path = _safe_output_path(out_path, overwrite=overwrite)

        print(
            f"  {label:5s} crop: "
            f"x0={clip.x0:.2f}, y0={clip.y0:.2f}, "
            f"x1={clip.x1:.2f}, y1={clip.y1:.2f} pt "
            f"-> {out_path.name}"
        )

        if dry_run:
            continue

        pix = page.get_pixmap(
            matrix=matrix,
            clip=clip,
            alpha=False,
            annots=True,
        )
        pix.save(str(out_path))
        created.append(out_path)

    doc.close()
    return created


def iter_pdfs(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        pdfs = sorted(input_dir.rglob(pattern))
    else:
        pdfs = sorted(input_dir.glob(pattern))

    return [p for p in pdfs if p.is_file() and p.suffix.lower() == ".pdf"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract front and back cover panels from full-wrap cover PDFs "
            "and save them as 8.5 x 11 PNG files."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="assets/book_covers",
        help="Folder containing cover PDFs. Default: assets/book_covers",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help='PDF filename pattern. Default: "*.pdf"',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output folder. If omitted, PNG files are saved next to each PDF."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --input-dir.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG render resolution. Default: 300",
    )
    parser.add_argument(
        "--trim-width",
        type=float,
        default=8.5,
        help="Final panel trim width in inches. Default: 8.5",
    )
    parser.add_argument(
        "--trim-height",
        type=float,
        default=11.0,
        help="Final panel trim height in inches. Default: 11.0",
    )
    parser.add_argument(
        "--bleed",
        type=float,
        default=0.125,
        help="Bleed to remove from outer/top/bottom edges in inches. Default: 0.125",
    )
    parser.add_argument(
        "--front-side",
        choices=["right", "left"],
        default="right",
        help=(
            "Which side of the full-wrap PDF contains the front cover. "
            "Default: right"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print crop information without writing PNG files.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_dir.exists():
        print(f"ERROR: input folder does not exist: {input_dir}", file=sys.stderr)
        return 1

    pdfs = iter_pdfs(input_dir, args.pattern, args.recursive)

    if not pdfs:
        print(
            f"No PDF files found in {input_dir} matching pattern {args.pattern!r}.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(pdfs)} PDF file(s).")

    total_created = 0

    for pdf_path in pdfs:
        try:
            created = extract_front_back_from_pdf(
                pdf_path,
                output_dir=output_dir,
                dpi=args.dpi,
                trim_width_in=args.trim_width,
                trim_height_in=args.trim_height,
                bleed_in=args.bleed,
                front_side=args.front_side,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            total_created += len(created)
        except Exception as exc:
            print(f"\nERROR processing {pdf_path}: {exc}", file=sys.stderr)

    if args.dry_run:
        print("\nDry run complete. No PNG files were written.")
    else:
        print(f"\nDone. Created {total_created} PNG file(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())