from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from python.publishing.qc.validate_public_technique_names import validate_public_technique_names

from python.publishing.distribution.imprint_registry import get_imprint_metadata
from python.publishing.distribution.isbn_registry import get_isbn_assignment
from python.publishing.distribution.kdp_profile_exporter import export_kdp_profile_json
from python.publishing.distribution.metadata_models import (
    DistributionPackageMetadata,
    ImprintMetadata,
    IsbnAssignment,
    MarketplaceMetadata,
)


def export_publication_metadata(
    *,
    publication_dir: Path,
    output_path: Path,
) -> Path:
    publication_manifest_path = publication_dir / "publication_manifest.json"
    cover_manifest_path = publication_dir / "cover_manifest.json"
    publication_package_path = publication_dir / "publication_package.json"
    publication_spec_path = _resolve_publication_spec_path(publication_dir)

    if not publication_manifest_path.exists():
        raise FileNotFoundError(f"Missing publication_manifest.json in {publication_dir}")

    publication_manifest = json.loads(publication_manifest_path.read_text(encoding="utf-8"))
    cover_manifest = None
    if cover_manifest_path.exists():
        cover_manifest = json.loads(cover_manifest_path.read_text(encoding="utf-8"))

    publication_spec = {}
    if publication_spec_path and publication_spec_path.exists():
        publication_spec = json.loads(publication_spec_path.read_text(encoding="utf-8"))

    metadata_block = publication_spec.get("metadata", {}) if publication_spec else {}

    imprint = _resolve_imprint(metadata_block)
    isbn = _resolve_isbn(metadata_block)
    marketplace = _resolve_marketplace(metadata_block, publication_manifest)

    distribution_metadata = DistributionPackageMetadata(
        publication_id=str(publication_manifest.get("publication_id", "")),
        book_id=str(publication_manifest.get("book_id", "")),
        title=str(publication_manifest.get("book_title", "")),
        subtitle=str(publication_manifest.get("book_subtitle", "")),
        imprint=imprint,
        isbn=isbn,
        marketplace=marketplace,
        extra={
            "library_id": publication_manifest.get("library_id"),
            "aisle_id": publication_manifest.get("aisle_id"),
            "channel_id": publication_manifest.get("channel_id"),
            "format_id": publication_manifest.get("format_id"),
            "binding_type": publication_manifest.get("binding_type"),
            "vendor": publication_manifest.get("vendor"),
            "trim_size": publication_manifest.get("trim_size"),
            "include_cover": publication_manifest.get("include_cover"),
            "include_solutions": publication_manifest.get("include_solutions"),
            "front_matter_profile": publication_manifest.get("front_matter_profile"),
            "end_matter_profile": publication_manifest.get("end_matter_profile"),
            "page_numbering_policy": publication_manifest.get("page_numbering_policy"),
            "puzzle_page_template": publication_manifest.get("puzzle_page_template"),
            "solution_page_template": publication_manifest.get("solution_page_template"),
            "cover_template": publication_manifest.get("cover_template"),
            "paper_type": publication_manifest.get("paper_type"),
            "estimated_page_count": publication_manifest.get("estimated_page_count"),
            "cover": {
                "cover_id": cover_manifest.get("cover_id") if cover_manifest else None,
                "spine_text": cover_manifest.get("spine_text") if cover_manifest else None,
                "author_imprint": cover_manifest.get("author_imprint") if cover_manifest else None,
                "isbn": cover_manifest.get("isbn") if cover_manifest else None,
                "geometry": cover_manifest.get("geometry") if cover_manifest else None,
            },
        },
    )

    payload = distribution_metadata.to_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


    public_technique_report = validate_public_technique_names(paths=[output_path])
    if int(public_technique_report.get("error_count") or 0) > 0:
        raise ValueError(
            "Public technique-name validation failed for distribution metadata: "
            f"{output_path}"
        )

    if marketplace and marketplace.marketplace == "amazon_kdp":
        export_kdp_profile_json(
            distribution_metadata=distribution_metadata,
            output_path=output_path.parent / "kdp_profile.json",
        )

    return output_path


def _resolve_publication_spec_path(publication_dir: Path) -> Path | None:
    package_path = publication_dir / "publication_package.json"
    if not package_path.exists():
        return None

    package = json.loads(package_path.read_text(encoding="utf-8"))
    publication_id = str(package.get("publication_id", ""))
    book_id = str(package.get("book_id", ""))

    publication_specs_dir = publication_dir.parent.parent / "publication_specs"
    if not publication_specs_dir.exists():
        return None

    for path in publication_specs_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("publication_id", "")) == publication_id and str(data.get("book_id", "")) == book_id:
            return path

    return None


def _resolve_imprint(metadata_block: Dict[str, Any]) -> ImprintMetadata | None:
    imprint_id = str(metadata_block.get("imprint_id", "")).strip()
    imprint_name = str(metadata_block.get("imprint_name", "")).strip()

    if imprint_id:
        return get_imprint_metadata(imprint_id)

    if imprint_name:
        return ImprintMetadata(
            imprint_id=imprint_name.lower().replace(" ", "_"),
            imprint_name=imprint_name,
            publisher_name=imprint_name,
        )

    return None


def _resolve_isbn(metadata_block: Dict[str, Any]) -> IsbnAssignment | None:
    isbn_assignment = str(metadata_block.get("isbn_assignment", "")).strip()
    isbn_direct = str(metadata_block.get("isbn", "")).strip()

    if isbn_assignment:
        return get_isbn_assignment(isbn_assignment)

    if isbn_direct:
        return IsbnAssignment(
            isbn13=isbn_direct,
            assignment_name="direct_from_publication_spec",
            status="active",
        )

    return None


def _resolve_marketplace(metadata_block: Dict[str, Any], publication_manifest: Dict[str, Any]) -> MarketplaceMetadata:
    return MarketplaceMetadata(
        marketplace=str(metadata_block.get("marketplace", "amazon_kdp")),
        language=str(metadata_block.get("language", "English")),
        description=str(
            metadata_block.get("marketplace_description")
            or metadata_block.get("back_copy")
            or publication_manifest.get("book_subtitle")
            or ""
        ),
        keywords=list(metadata_block.get("keywords", [])),
        categories=list(metadata_block.get("categories", [])),
        audience=str(metadata_block.get("audience", "General")),
        contributor_name=str(metadata_block.get("contributor_name", "")),
        series_name=str(metadata_block.get("series_name") or publication_manifest.get("book_title") or ""),
    )