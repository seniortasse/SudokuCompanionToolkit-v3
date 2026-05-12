from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from python.publishing.book_builder.book_spec_loader import BookSectionSpec
from python.publishing.book_builder.dedupe import filter_reusable_puzzles
from python.publishing.book_builder.distribution_balancer import apply_distribution_constraints
from python.publishing.book_builder.ordering import order_section_puzzles
from python.publishing.book_builder.puzzle_selector import select_puzzles_for_section
from python.publishing.book_builder.random_selector import apply_seeded_shuffle
from python.publishing.book_builder.selection_priority import apply_selection_priority
from python.publishing.ids.id_policy import build_section_id
from python.publishing.inventory.assignment_rules import filter_records_available_for_library
from python.publishing.schemas.models import PatternRecord, PuzzleRecord, SectionCriteria, SectionManifest


@dataclass
class AllocatedSection:
    section_spec: BookSectionSpec
    section_manifest: SectionManifest
    assigned_puzzles: List[PuzzleRecord]


def allocate_sections(
    *,
    book_id: str,
    puzzle_records: List[PuzzleRecord],
    section_specs: List[BookSectionSpec],
    ordering_policy: dict,
    reuse_policy: str,
    global_filters: Dict | None = None,
    library_inventory: dict | None = None,
    pattern_records_by_id: Optional[Mapping[str, PatternRecord]] = None,
) -> List[AllocatedSection]:
    allocated: List[AllocatedSection] = []
    used_record_ids: set[str] = set()

    base_records = list(puzzle_records)
    if library_inventory is not None:
        base_records = filter_records_available_for_library(
            records=base_records,
            inventory=library_inventory,
        )

    effective_global_filters = dict(global_filters or {})

    for section_order, section_spec in enumerate(section_specs, start=1):
        criteria = dict(section_spec.criteria or {})

        candidates = select_puzzles_for_section(
            puzzle_records=base_records,
            section_criteria=criteria,
            global_filters=effective_global_filters,
            pattern_records_by_id=pattern_records_by_id,
        )

        candidates = filter_reusable_puzzles(
            candidates,
            blocklist=used_record_ids,
            reuse_policy=reuse_policy,
        )

        candidates = apply_seeded_shuffle(
            candidates,
            random_seed=criteria.get("random_seed"),
        )

        candidates = apply_selection_priority(
            records=candidates,
            section_criteria=criteria,
        )

        candidates = apply_distribution_constraints(
            records=candidates,
            target_count=section_spec.puzzle_count,
            min_distinct_patterns=criteria.get("min_distinct_patterns"),
            max_distinct_patterns=criteria.get("max_distinct_patterns"),
            min_distinct_pattern_families=criteria.get("min_distinct_pattern_families"),
            max_distinct_pattern_families=criteria.get("max_distinct_pattern_families"),
            pattern_occurrence_caps=criteria.get("pattern_occurrence_caps"),
            pattern_family_occurrence_caps=criteria.get("pattern_family_occurrence_caps"),
        )

        candidates = order_section_puzzles(
            candidates,
            ordering_policy=ordering_policy,
        )

        assigned = candidates[: section_spec.puzzle_count]
        if len(assigned) < section_spec.puzzle_count:
            raise ValueError(
                f"Section {section_spec.section_code} requested {section_spec.puzzle_count} puzzles "
                f"but only {len(assigned)} eligible puzzles were available after applying "
                f"criteria, pattern-signal filters, inventory exclusions, reuse policy, and distribution constraints"
            )

        used_record_ids.update(record.record_id for record in assigned)

        section_manifest = SectionManifest(
            section_id=build_section_id(section_spec.section_code),
            book_id=book_id,
            section_code=section_spec.section_code,
            title=section_spec.title,
            subtitle=section_spec.subtitle,
            section_order=section_order,
            criteria=SectionCriteria.from_dict(criteria),
            difficulty_label_hint=section_spec.difficulty_label_hint,
            puzzle_count=len(assigned),
            puzzle_ids=[record.record_id for record in assigned],
        )

        allocated.append(
            AllocatedSection(
                section_spec=section_spec,
                section_manifest=section_manifest,
                assigned_puzzles=assigned,
            )
        )

    return allocated