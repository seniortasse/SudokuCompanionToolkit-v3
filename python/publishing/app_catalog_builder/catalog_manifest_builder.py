from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from python.publishing.ids.id_policy import build_aisle_id
from python.publishing.schemas.models import (
    AisleManifest,
    BookSummary,
    CatalogManifest,
    LibraryManifest,
    PuzzleRecord,
)


@dataclass(frozen=True)
class AisleDefinition:
    aisle_id: str
    slug: str
    title: str
    description: str
    sort_order: int
    organization_principle: str


DEFAULT_CLASSIC9_AISLES: List[AisleDefinition] = [
    AisleDefinition(
        aisle_id=build_aisle_id("TCOUNT"),
        slug="technique-count",
        title="Technique Count",
        description="Books organized by the number of techniques used to solve puzzles.",
        sort_order=1,
        organization_principle="technique_count",
    ),
    AisleDefinition(
        aisle_id=build_aisle_id("CTECH"),
        slug="custom-technique",
        title="Custom Technique",
        description="Books organized around specific techniques such as X-Wing or Y-Wing.",
        sort_order=2,
        organization_principle="featured_technique",
    ),
    AisleDefinition(
        aisle_id=build_aisle_id("DWEIGHT"),
        slug="difficulty-weight",
        title="Puzzle Difficulty",
        description="Books organized by puzzle effort and weight bands.",
        sort_order=3,
        organization_principle="weight_range",
    ),
    AisleDefinition(
        aisle_id=build_aisle_id("PATTERN"),
        slug="custom-pattern",
        title="Custom Pattern",
        description="Books organized around visual clue patterns and pattern families.",
        sort_order=4,
        organization_principle="pattern_family",
    ),
    AisleDefinition(
        aisle_id=build_aisle_id("MISC"),
        slug="miscellaneous",
        title="Miscellaneous",
        description="Books with custom or mixed organizational strategies.",
        sort_order=5,
        organization_principle="custom",
    ),
]


def build_library_manifest(
    *,
    library_id: str,
    slug: str,
    title: str,
    subtitle: str,
    description: str,
    layout_type: str,
    grid_size: int,
    charset: str,
    box_schema: str,
    status: str,
    aisle_definitions: Iterable[AisleDefinition],
    created_at: str | None = None,
    updated_at: str | None = None,
) -> LibraryManifest:
    aisle_ids = [definition.aisle_id for definition in aisle_definitions]
    return LibraryManifest(
        library_id=library_id,
        slug=slug,
        title=title,
        subtitle=subtitle,
        description=description,
        layout_type=layout_type,
        grid_size=grid_size,
        charset=charset,
        box_schema=box_schema,
        status=status,
        aisle_ids=aisle_ids,
        created_at=created_at,
        updated_at=updated_at,
    )


def build_aisle_manifests(
    *,
    library_id: str,
    puzzle_records: Iterable[PuzzleRecord],
    aisle_definitions: Iterable[AisleDefinition] = DEFAULT_CLASSIC9_AISLES,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> List[AisleManifest]:
    records = list(puzzle_records)
    manifests: List[AisleManifest] = []

    for definition in aisle_definitions:
        book_ids = sorted(
            {
                record.book_id
                for record in records
                if record.aisle_id == definition.aisle_id and record.book_id
            }
        )

        manifests.append(
            AisleManifest(
                aisle_id=definition.aisle_id,
                library_id=library_id,
                slug=definition.slug,
                title=definition.title,
                description=definition.description,
                sort_order=definition.sort_order,
                organization_principle=definition.organization_principle,
                book_ids=book_ids,
                created_at=created_at,
                updated_at=updated_at,
            )
        )

    return manifests


def _build_book_summaries(puzzle_records: Iterable[PuzzleRecord]) -> List[BookSummary]:
    by_book: Dict[str, Dict[str, object]] = {}

    for record in puzzle_records:
        if not record.book_id:
            continue

        entry = by_book.setdefault(
            record.book_id,
            {
                "book_id": record.book_id,
                "title": record.book_id,
                "subtitle": "",
                "aisle_id": record.aisle_id,
                "puzzle_count": 0,
            },
        )
        entry["puzzle_count"] = int(entry["puzzle_count"]) + 1

    summaries = [
        BookSummary(
            book_id=str(entry["book_id"]),
            title=str(entry["title"]),
            subtitle=str(entry["subtitle"]),
            aisle_id=str(entry["aisle_id"]),
            puzzle_count=int(entry["puzzle_count"]),
        )
        for entry in by_book.values()
    ]

    summaries.sort(key=lambda x: x.book_id)
    return summaries


def build_catalog_manifest(
    *,
    catalog_version: str,
    generated_at: str,
    library_manifest: LibraryManifest,
    puzzle_records: Iterable[PuzzleRecord],
    index_files: Dict[str, str],
) -> CatalogManifest:
    records = list(puzzle_records)
    return CatalogManifest(
        catalog_version=catalog_version,
        generated_at=generated_at,
        library_ids=[library_manifest.library_id],
        libraries=[library_manifest],
        book_summaries=_build_book_summaries(records),
        index_files=dict(index_files),
    )