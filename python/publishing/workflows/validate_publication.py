from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from python.publishing.qc.validate_public_technique_names import validate_public_technique_names
from python.publishing.workflows.publication_package_reader import (
    is_page_type,
    load_publication_artifacts,
    normalize_page_type,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a built publication package and emit a QC summary."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory.",
    )
    return parser.parse_args()


def _first_title_page_block(interior_plan: dict) -> dict:
    for block in list(interior_plan.get("page_blocks") or []):
        if normalize_page_type(block.get("page_type")) == "title_page":
            return dict(block)
    return {}


def _clean_text(value) -> str:
    return " ".join(str(value or "").strip().split())


def validate_publication_dir(*, publication_dir: Path) -> Dict[str, Any]:
    if not publication_dir.exists():
        raise FileNotFoundError(f"publication directory not found: {publication_dir}")

    from python.publishing.print_specs import get_print_format_spec, validate_print_format_spec

    required_files = [
        "publication_manifest.json",
        "publication_package.json",
        "interior_plan.json",
    ]

    errors = []
    warnings = []

    for name in required_files:
        if not (publication_dir / name).exists():
            errors.append(f"Missing required file: {name}")

    try:
        artifacts = load_publication_artifacts(publication_dir)
        publication_manifest = artifacts["manifest"]
        interior_plan = artifacts["interior"]
        cover_manifest = artifacts["cover_manifest"]
    except Exception as exc:
        errors.append(f"Unable to parse publication JSON files: {exc}")
        publication_manifest = {}
        interior_plan = {}
        cover_manifest = {}

    format_id = publication_manifest.get("format_id")
    if format_id:
        try:
            format_spec = get_print_format_spec(format_id)
            errors.extend(validate_print_format_spec(format_spec))
        except Exception as exc:
            errors.append(f"Invalid format reference '{format_id}': {exc}")

    metadata = dict(publication_manifest.get("metadata") or {})
    kdp_listing = dict(metadata.get("kdp_listing") or {})

    expected_title = _clean_text(kdp_listing.get("title") or metadata.get("title"))
    expected_subtitle = _clean_text(kdp_listing.get("subtitle") or metadata.get("subtitle"))

    title_page = _first_title_page_block(interior_plan)
    title_payload = dict(title_page.get("payload") or {})

    rendered_title = _clean_text(title_payload.get("title"))
    rendered_subtitle = _clean_text(title_payload.get("subtitle"))

    if expected_title and rendered_title and expected_title != rendered_title:
        errors.append(
            "KDP title mismatch: "
            f"metadata.kdp_listing.title={expected_title!r} but title page payload title={rendered_title!r}"
        )

    if expected_subtitle and rendered_subtitle and expected_subtitle != rendered_subtitle:
        errors.append(
            "KDP subtitle mismatch: "
            f"metadata.kdp_listing.subtitle={expected_subtitle!r} but title page payload subtitle={rendered_subtitle!r}"
        )

    interior_bleed_mode = str(publication_manifest.get("interior_bleed_mode") or "both").strip().lower()
    if interior_bleed_mode not in {"no_bleed", "bleed", "both"}:
        errors.append(
            f"Invalid interior_bleed_mode={interior_bleed_mode!r}. "
            "Use 'no_bleed', 'bleed', or 'both'."
        )

    if str(publication_manifest.get("channel_id") or "").lower().startswith("amazon"):
        if interior_bleed_mode == "no_bleed":
            warnings.append(
                "Amazon/KDP publication is configured for no_bleed interior export. "
                "Use interior_bleed_mode='bleed' or 'both' if the KDP Bookshelf bleed setting is enabled."
            )



    page_blocks = interior_plan.get("page_blocks", []) or []
    estimated_page_count = int(interior_plan.get("estimated_page_count", 0))
    if estimated_page_count != len(page_blocks):
        warnings.append(
            f"estimated_page_count={estimated_page_count} does not match page_blocks={len(page_blocks)}"
        )

    for idx, block in enumerate(page_blocks, start=1):
        page_index = block.get("page_index")
        physical_page_number = block.get("physical_page_number")
        logical_page_number = block.get("logical_page_number")
        page_type = normalize_page_type(block.get("page_type"))

        if page_index != idx:
            warnings.append(f"Page block #{idx} has page_index={page_index}")

        if physical_page_number != idx:
            warnings.append(
                f"Page block #{idx} has physical_page_number={physical_page_number}"
            )

        if logical_page_number in (None, 0):
            warnings.append(
                f"Page block #{idx} is missing logical_page_number"
            )

        show_page_number = bool(block.get("show_page_number", False))
        if page_type == "blank_page" and show_page_number:
            errors.append(f"Blank page #{idx} incorrectly shows a page number")

    include_solutions = bool(publication_manifest.get("include_solutions", True))
    has_solution_pages = any(is_page_type(block.get("page_type"), "solution_page") for block in page_blocks)

    if include_solutions and not has_solution_pages:
        errors.append("Manifest requests solutions but interior plan has no solution pages")

    if not include_solutions and has_solution_pages:
        errors.append("Manifest disables solutions but interior plan still contains solution pages")

    layout = dict(publication_manifest.get("layout_config") or {})

    puzzles_per_page = layout.get("puzzles_per_page")

    solution_puzzles_per_page = (
        layout.get("solution_puzzles_per_page")
        or (
            int(layout.get("solution_rows")) * int(layout.get("solution_cols"))
            if layout.get("solution_rows") is not None
            and layout.get("solution_cols") is not None
            else None
        )
        or puzzles_per_page
    )

    for idx, block in enumerate(page_blocks, start=1):
        payload = dict(block.get("payload") or {})
        puzzle_ids = list(payload.get("puzzle_ids") or [])

        if not puzzle_ids:
            continue

        if is_page_type(block.get("page_type"), "puzzle_page"):
            if puzzles_per_page is None:
                continue

            max_per_page = int(puzzles_per_page)
            if len(puzzle_ids) > max_per_page:
                errors.append(
                    f"Page block #{idx} exceeds puzzles_per_page: {len(puzzle_ids)} > {max_per_page}"
                )

        elif is_page_type(block.get("page_type"), "solution_page"):
            page_solution_limit = (
                payload.get("solution_puzzles_per_page")
                or solution_puzzles_per_page
            )

            if page_solution_limit is None:
                continue

            max_per_page = int(page_solution_limit)
            if len(puzzle_ids) > max_per_page:
                errors.append(
                    f"Page block #{idx} exceeds solution_puzzles_per_page: {len(puzzle_ids)} > {max_per_page}"
                )

    if cover_manifest:
        geometry = cover_manifest.get("geometry", {})
        if float(geometry.get("spine_width_in", 0.0)) < 0:
            errors.append("Cover geometry has negative spine_width_in")

    
    public_technique_report = validate_public_technique_names(paths=[publication_dir])
    for violation in list(public_technique_report.get("errors") or []):
        errors.append(
            "Public technique-name leak: "
            f"{violation.get('path')}:{violation.get('line_number')} "
            f"matched {violation.get('pattern')} -> {violation.get('line')}"
        )

    summary = {
        "publication_dir": str(publication_dir),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }

    summary_path = publication_dir / "qc_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> int:
    args = _parse_args()
    publication_dir = Path(args.publication_dir)

    _log("=" * 72)
    _log("validate_publication.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    try:
        summary = validate_publication_dir(publication_dir=publication_dir)
    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1

    errors = list(summary.get("errors") or [])
    warnings = list(summary.get("warnings") or [])
    summary_path = publication_dir / "qc_summary.json"

    if errors:
        _log("VALIDATION FAILED")
        _log("-" * 72)
        for err in errors:
            _log(f"- {err}")
    else:
        _log("VALIDATION PASSED")

    if warnings:
        _log("-" * 72)
        for warning in warnings:
            _log(f"WARNING: {warning}")

    _log("-" * 72)
    _log(f"QC summary written to: {summary_path}")
    _log("=" * 72)
    _log("validate_publication.py completed")
    _log("=" * 72)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())