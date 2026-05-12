from __future__ import annotations

from typing import Any, Dict

from python.publishing.cover_builder.cover_geometry_builder import build_cover_geometry
from python.publishing.schemas.models import CoverSpec, PrintFormatSpec, PublicationSpec


def build_cover_manifest(
    *,
    publication_spec: PublicationSpec,
    format_spec: PrintFormatSpec,
    cover_spec: CoverSpec,
) -> Dict[str, Any]:
    geometry = build_cover_geometry(
        format_spec=format_spec,
        cover_spec=cover_spec,
    )

    return {
        "cover_id": cover_spec.cover_id,
        "publication_id": cover_spec.publication_id,
        "format_id": cover_spec.format_id,
        "cover_template": publication_spec.cover_template,
        "paper_type": cover_spec.paper_type,
        "page_count": cover_spec.page_count,
        "spine_text": cover_spec.spine_text,
        "back_copy": cover_spec.back_copy,
        "author_imprint": cover_spec.author_imprint,
        "isbn": cover_spec.isbn,
        "front_design_asset": cover_spec.front_design_asset,
        "back_design_asset": cover_spec.back_design_asset,
        "geometry": geometry,
    }