from __future__ import annotations

from typing import Any, Dict

from python.publishing.schemas.publication_manifest import PublicationManifest


def _variant_identity_block(*, book_manifest, publication_spec, format_spec) -> Dict[str, Any]:
    layout = publication_spec.layout_config
    metadata = dict(publication_spec.metadata or {})

    language_code = str(
        layout.language
        or metadata.get("locale")
        or metadata.get("language")
        or "en"
    )

    format_id = str(publication_spec.format_id or "")
    format_id_lower = format_id.lower()

    trim_token = (
        "8511"
        if format_id_lower in {
            "kdp_8_5x11_bw",
            "kdp_8_5x11_color",
            "kdp_8_5x11",
            "amazon_kdp_paperback_8_5x11_bw",
            "amazon_kdp_paperback_8_5x11_color",
        }
        else format_id
    )

    interior_bleed_mode = str(
        getattr(publication_spec, "interior_bleed_mode", "both") or "both"
    )

    return {
        "publication_id": publication_spec.publication_id,
        "book_id": book_manifest.book_id,
        "channel_id": publication_spec.channel_id,
        "format_id": publication_spec.format_id,
        "trim_width_in": format_spec.trim_width_in,
        "trim_height_in": format_spec.trim_height_in,
        "trim_token": trim_token,
        "paper_type": publication_spec.paper_type,
        "interior_bleed_mode": interior_bleed_mode,
        "interior_bleed_in": float(format_spec.bleed_in or 0.0),
        "language": str(metadata.get("language") or language_code),
        "language_code": language_code,
        "puzzles_per_page": layout.puzzles_per_page,
        "rows": layout.rows,
        "cols": layout.cols,
        "puzzle_page_template": publication_spec.puzzle_page_template,
        "solution_page_template": publication_spec.solution_page_template,
    }


def build_publication_manifest(
    *,
    book_manifest,
    publication_spec,
    format_spec,
    interior_plan,
    cover_spec,
):
    interior_bleed_mode = str(
        getattr(publication_spec, "interior_bleed_mode", "both") or "both"
    )

    return PublicationManifest(
        publication_id=publication_spec.publication_id,
        book_id=book_manifest.book_id,
        channel_id=publication_spec.channel_id,
        format_id=publication_spec.format_id,
        trim_width_in=format_spec.trim_width_in,
        trim_height_in=format_spec.trim_height_in,
        paper_type=publication_spec.paper_type,
        include_cover=publication_spec.include_cover,
        include_solutions=publication_spec.include_solutions,
        mirror_margins=bool(publication_spec.mirror_margins),
        page_numbering_policy=publication_spec.page_numbering_policy,
        blank_page_policy=publication_spec.blank_page_policy,
        puzzle_page_template=publication_spec.puzzle_page_template,
        solution_page_template=publication_spec.solution_page_template,
        cover_template=publication_spec.cover_template,
        estimated_page_count=interior_plan.estimated_page_count,
        interior_bleed_mode=interior_bleed_mode,
        interior_bleed_in=float(format_spec.bleed_in or 0.0),
        cover_id=(cover_spec.cover_id if cover_spec is not None else None),
        metadata=dict(publication_spec.metadata or {}),
        layout_config=publication_spec.layout_config.to_dict(),
        solution_section_config=dict(
            getattr(publication_spec, "solution_section_config", {}) or {}
        ),
        variant_identity=_variant_identity_block(
            book_manifest=book_manifest,
            publication_spec=publication_spec,
            format_spec=format_spec,
        ),
    )