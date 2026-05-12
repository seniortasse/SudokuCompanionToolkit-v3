from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from python.publishing.book_builder.book_spec_loader import BookSpec
from python.publishing.book_builder.section_allocator import AllocatedSection, allocate_sections
from python.publishing.ids.id_policy import build_local_puzzle_code, build_puzzle_uid_for_book
from python.publishing.inventory.assignment_ledger import register_assignment
from python.publishing.inventory.library_inventory_store import (
    load_library_inventory,
    save_library_inventory,
)
from python.publishing.pattern_library.pattern_store import load_pattern_store
from python.publishing.schemas.models import BookManifest, PatternRecord, PuzzleRecord, SectionManifest


@dataclass
class BuiltBook:
    book_manifest: BookManifest
    section_manifests: List[SectionManifest]
    assigned_puzzles: List[PuzzleRecord]
    inventory_path: str | None = None


def _retag_puzzle_for_book(
    record: PuzzleRecord,
    *,
    book_id: str,
    aisle_id: str,
    section_id: str,
    section_code: str,
    position_in_section: int,
    position_in_book: int,
) -> PuzzleRecord:
    cloned = copy.deepcopy(record)
    local_puzzle_code = build_local_puzzle_code(section_code, position_in_section)
    cloned.aisle_id = aisle_id
    cloned.book_id = book_id
    cloned.section_id = section_id
    cloned.section_code = section_code
    cloned.local_puzzle_code = local_puzzle_code
    cloned.friendly_puzzle_id = f"{book_id}-{local_puzzle_code}"
    cloned.puzzle_uid = build_puzzle_uid_for_book(
        book_id=book_id,
        section_code=section_code,
        ordinal=position_in_section,
    )
    cloned.position_in_section = position_in_section
    cloned.position_in_book = position_in_book
    cloned.candidate_status = "assigned"
    cloned.print_header.display_code = local_puzzle_code
    return cloned


def _load_active_pattern_records_by_id(
    patterns_dir: str | Path | None,
) -> Optional[Mapping[str, PatternRecord]]:
    if patterns_dir is None:
        return None

    patterns_path = Path(patterns_dir)
    if not patterns_path.exists():
        return None

    registry = load_pattern_store(patterns_path)
    active_patterns = [pattern for pattern in registry.patterns if pattern.status == "active"]
    return {str(pattern.pattern_id): pattern for pattern in active_patterns}


def build_book_from_spec(
    *,
    spec: BookSpec,
    puzzle_records: List[PuzzleRecord],
    inventory_dir: str | Path | None = None,
    patterns_dir: str | Path | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> BuiltBook:
    inventory_base_dir = Path(inventory_dir) if inventory_dir is not None else None
    library_inventory = None
    if inventory_base_dir is not None:
        library_inventory = load_library_inventory(
            base_dir=inventory_base_dir,
            library_id=spec.library_id,
        )

    pattern_records_by_id = _load_active_pattern_records_by_id(patterns_dir)

    allocated_sections = allocate_sections(
        book_id=spec.book_id,
        puzzle_records=puzzle_records,
        section_specs=spec.sections,
        ordering_policy=spec.ordering_policy,
        reuse_policy=spec.reuse_policy,
        global_filters=spec.global_filters,
        library_inventory=library_inventory,
        pattern_records_by_id=pattern_records_by_id,
    )

    section_manifests: List[SectionManifest] = []
    assigned_puzzles: List[PuzzleRecord] = []
    position_in_book = 0

    for allocated in allocated_sections:
        section_manifest = allocated.section_manifest
        section_manifests.append(section_manifest)

        section_assigned_ids: List[str] = []

        for position_in_section, record in enumerate(allocated.assigned_puzzles, start=1):
            position_in_book += 1
            placed = _retag_puzzle_for_book(
                record,
                book_id=spec.book_id,
                aisle_id=spec.aisle_id,
                section_id=section_manifest.section_id,
                section_code=section_manifest.section_code,
                position_in_section=position_in_section,
                position_in_book=position_in_book,
            )
            assigned_puzzles.append(placed)
            if placed.puzzle_uid is not None:
                section_assigned_ids.append(placed.puzzle_uid)

        section_manifest.puzzle_ids = section_assigned_ids
        section_manifest.puzzle_count = len(section_assigned_ids)

    book_manifest = BookManifest(
        book_id=spec.book_id,
        library_id=spec.library_id,
        aisle_id=spec.aisle_id,
        title=spec.title,
        subtitle=spec.subtitle,
        series_name=spec.series_name,
        volume_number=spec.volume_number,
        isbn=spec.isbn,
        description=spec.description,
        target_audience=spec.target_audience,
        trim_size=spec.trim_size,
        puzzles_per_page=spec.puzzles_per_page,
        page_layout_profile=spec.page_layout_profile,
        solution_section_policy=spec.solution_section_policy,
        cover_theme=spec.cover_theme,
        layout_type=spec.layout_type,
        grid_size=spec.grid_size,
        section_ids=[section.section_id for section in section_manifests],
        puzzle_count=len(assigned_puzzles),
        publication_status=spec.publication_status,
        search_tags=list(spec.search_tags),
        created_at=created_at,
        updated_at=updated_at,
    )

    inventory_path: str | None = None
    if inventory_base_dir is not None and library_inventory is not None:
        for record in assigned_puzzles:
            register_assignment(
                library_inventory,
                record=record,
            )
        written_inventory_path = save_library_inventory(
            inventory=library_inventory,
            base_dir=inventory_base_dir,
        )
        inventory_path = str(written_inventory_path)

    return BuiltBook(
        book_manifest=book_manifest,
        section_manifests=section_manifests,
        assigned_puzzles=assigned_puzzles,
        inventory_path=inventory_path,
    )