from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from python.publishing.qc.validate_public_technique_names import (
    validate_public_technique_names,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a full publication bundle containing interior PDF, cover PDF, "
            "metadata, previews, and manifests."
        )
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the output bundle directory.",
    )
    parser.add_argument(
        "--interior-filename",
        default="interior.pdf",
        help="Filename to use for the preferred exported interior PDF.",
    )
    parser.add_argument(
        "--interior-bleed-mode",
        default=None,
        choices=["no_bleed", "bleed", "both"],
        help=(
            "Interior export mode. Defaults to publication_manifest.interior_bleed_mode "
            "or 'both'."
        ),
    )
    parser.add_argument(
        "--cover-filename",
        default="cover.pdf",
        help="Filename to use for the exported cover PDF.",
    )
    return parser.parse_args()


def _resolve_existing_or_parent(path: Path) -> Path:
    """
    Resolve a path even when it does not exist yet.

    Path.resolve(strict=False) is available on modern Python, but this helper keeps
    the intent explicit and gives us a stable absolute path for safety checks.
    """
    return path.expanduser().resolve(strict=False)


def _assert_safe_output_dir(*, publication_dir: Path, output_dir: Path) -> None:
    """
    Prevent accidental deletion of broad/project directories.

    The bundle exporter intentionally deletes output_dir before regenerating it.
    That is safe only if output_dir is a specific bundle leaf directory, not a
    parent folder such as exports/, bundles/, the project root, or the source
    publication directory.
    """
    publication_resolved = _resolve_existing_or_parent(publication_dir)
    output_resolved = _resolve_existing_or_parent(output_dir)

    if output_resolved == publication_resolved:
        raise ValueError(
            "Refusing to clean output_dir because it is the same as publication_dir: "
            f"{output_resolved}"
        )

    if output_resolved in publication_resolved.parents:
        raise ValueError(
            "Refusing to clean output_dir because it is a parent of publication_dir: "
            f"{output_resolved}"
        )

    # Refuse suspiciously broad/common directory names. The expected output path is
    # normally exports/sudoku_books/bundles/<BOOK_ID>__<PUBLICATION_ID>.
    dangerous_names = {
        "",
        ".",
        "/",
        "\\",
        "exports",
        "export",
        "bundles",
        "bundle",
        "sudoku_books",
        "classic9",
        "publications",
        "publication_specs",
        "publication_specs_compiled",
        "datasets",
        "runs",
        "python",
        "src",
        "SudokuCompanionToolkit v3",
    }

    if output_resolved.name in dangerous_names:
        raise ValueError(
            "Refusing to clean suspiciously broad output_dir. "
            "Pass a specific bundle directory instead: "
            f"{output_resolved}"
        )

    # Your bundle directory convention contains "__" between book id and
    # publication id. Do not make this mandatory for compatibility, but warn via
    # exception only when the path is clearly too shallow.
    if len(output_resolved.parts) < 4:
        raise ValueError(
            "Refusing to clean output_dir because the path is too shallow: "
            f"{output_resolved}"
        )


def _clean_output_dir(*, publication_dir: Path, output_dir: Path) -> None:
    """
    Export into a clean bundle directory every time.

    This prevents stale files from previous exports from causing false failures,
    especially:
      - old interior.pdf / interior_bleed.pdf / interior_no_bleed.pdf
      - old public_technique_name_qc.json
      - old preview images
      - old generated cover assets
      - old manifests copied from a previous variant
    """
    _assert_safe_output_dir(publication_dir=publication_dir, output_dir=output_dir)

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output_dir exists but is not a directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def export_publication_bundle(
    *,
    publication_dir: Path,
    output_dir: Path,
    interior_filename: str = "interior.pdf",
    cover_filename: str = "cover.pdf",
    interior_bleed_mode: str | None = None,
) -> Dict[str, Any]:
    if not publication_dir.exists():
        raise FileNotFoundError(f"publication directory not found: {publication_dir}")

    # Always export into a clean bundle directory.
    #
    # This is the long-term stale-output fix. Without it, old JSON/PDF/preview
    # files from an earlier failed or outdated export can remain in the bundle
    # folder and make the current clean export appear broken.
    _clean_output_dir(publication_dir=publication_dir, output_dir=output_dir)

    previews_dir = output_dir / "previews"
    interior_pdf_path = output_dir / interior_filename
    cover_pdf_path = output_dir / cover_filename

    from python.publishing.pdf_renderer.compat import patch_hashlib_usedforsecurity

    patch_hashlib_usedforsecurity()

    from python.publishing.distribution import (
        build_asset_manifest,
        export_publication_metadata,
        export_publication_previews,
    )
    from python.publishing.pdf_renderer.cover_pdf_exporter import export_book_cover_pdf
    from python.publishing.pdf_renderer.interior_pdf_exporter import export_book_interior_pdf

    warnings: list[str] = []

    manifest_path = publication_dir / "publication_manifest.json"
    publication_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    requested_mode = str(
        interior_bleed_mode
        or publication_manifest.get("interior_bleed_mode")
        or "both"
    ).strip().lower()

    if requested_mode not in {"no_bleed", "bleed", "both"}:
        raise ValueError(
            f"Unsupported interior_bleed_mode={requested_mode!r}. "
            "Use 'no_bleed', 'bleed', or 'both'."
        )

    interior_no_bleed_path = output_dir / "interior_no_bleed.pdf"
    interior_bleed_path = output_dir / "interior_bleed.pdf"

    if requested_mode in {"no_bleed", "both"}:
        export_book_interior_pdf(
            publication_dir=publication_dir,
            output_pdf_path=interior_no_bleed_path,
            bleed_mode="no_bleed",
        )

    if requested_mode in {"bleed", "both"}:
        export_book_interior_pdf(
            publication_dir=publication_dir,
            output_pdf_path=interior_bleed_path,
            bleed_mode="bleed",
        )

    # Preferred canonical name:
    # - for "bleed" and "both", interior.pdf is Amazon/KDP-ready.
    # - for "no_bleed", interior.pdf is trim-size.
    preferred_source = (
        interior_bleed_path
        if requested_mode in {"bleed", "both"}
        else interior_no_bleed_path
    )

    if not preferred_source.exists():
        raise FileNotFoundError(
            f"Preferred interior PDF was not produced: {preferred_source}"
        )

    shutil.copy2(preferred_source, interior_pdf_path)

    cover_manifest_exists = (publication_dir / "cover_manifest.json").exists()
    if cover_manifest_exists:
        export_book_cover_pdf(
            publication_dir=publication_dir,
            output_pdf_path=cover_pdf_path,
        )

    copied_names = [
        "publication_manifest.json",
        "publication_package.json",
        "interior_plan.json",
        "cover_spec.json",
        "cover_manifest.json",
        "qc_summary.json",
        "public_technique_name_qc.json",
    ]
    for name in copied_names:
        src = publication_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    export_publication_metadata(
        publication_dir=publication_dir,
        output_path=output_dir / "metadata.json",
    )

    preview_results: Dict[str, Any] = {}
    try:
        preview_results = export_publication_previews(
            publication_dir=publication_dir,
            output_dir=previews_dir,
        )
    except Exception as exc:
        warnings.append(f"Preview export skipped: {exc}")

    asset_manifest = build_asset_manifest(bundle_dir=output_dir)
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(asset_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    public_technique_report = validate_public_technique_names(paths=[output_dir])
    (output_dir / "public_technique_name_qc.json").write_text(
        json.dumps(public_technique_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if int(public_technique_report.get("error_count") or 0) > 0:
        raise ValueError(
            "Public technique-name validation failed for exported bundle. "
            f"See {output_dir / 'public_technique_name_qc.json'}"
        )

    summary = {
        "publication_dir": str(publication_dir),
        "output_dir": str(output_dir),
        "interior_pdf": str(interior_pdf_path),
        "interior_no_bleed_pdf": (
            str(interior_no_bleed_path) if interior_no_bleed_path.exists() else None
        ),
        "interior_bleed_pdf": (
            str(interior_bleed_path) if interior_bleed_path.exists() else None
        ),
        "interior_bleed_mode": requested_mode,
        "cover_pdf": str(cover_pdf_path) if cover_manifest_exists else None,
        "previews": preview_results,
        "warnings": warnings,
    }
    (output_dir / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> int:
    args = _parse_args()

    publication_dir = Path(args.publication_dir)
    output_dir = Path(args.output_dir)

    _log("=" * 72)
    _log("export_publication_bundle.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log(f"Output dir:      {output_dir.resolve()}")
    _log("=" * 72)

    try:
        summary = export_publication_bundle(
            publication_dir=publication_dir,
            output_dir=output_dir,
            interior_filename=args.interior_filename,
            cover_filename=args.cover_filename,
            interior_bleed_mode=args.interior_bleed_mode,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"reportlab", "rlPyCairo", "_renderPM"}:
            _log("ERROR: Missing ReportLab preview/render dependencies.")
            _log("Install them in your active virtual environment with:")
            _log("  python -m pip install reportlab")
            return 1
        _log(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1

    _log(f"Interior PDF:   {summary['interior_pdf']}")

    interior_no_bleed_pdf = summary.get("interior_no_bleed_pdf")
    interior_bleed_pdf = summary.get("interior_bleed_pdf")
    if interior_no_bleed_pdf:
        _log(f"No-bleed PDF:   {interior_no_bleed_pdf}")
    if interior_bleed_pdf:
        _log(f"Bleed PDF:      {interior_bleed_pdf}")

    if summary.get("cover_pdf"):
        _log(f"Cover PDF:      {summary['cover_pdf']}")
    else:
        _log("Cover PDF:      skipped (no cover_manifest.json present)")

    _log(f"Metadata:       {output_dir / 'metadata.json'}")

    preview_results = dict(summary.get("previews") or {})
    if preview_results:
        _log(f"Previews dir:   {output_dir / 'previews'}")
    else:
        _log("Previews dir:   skipped")

    for warning in list(summary.get("warnings") or []):
        _log(f"WARNING:        {warning}")

    _log(f"Assets:         {output_dir / 'asset_manifest.json'}")
    _log(f"Bundle summary: {output_dir / 'bundle_summary.json'}")
    _log("=" * 72)
    _log("export_publication_bundle.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())