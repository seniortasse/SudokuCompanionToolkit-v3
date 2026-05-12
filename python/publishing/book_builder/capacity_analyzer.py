from __future__ import annotations

from typing import Dict, List

from python.publishing.book_builder.puzzle_selector import select_puzzles_for_section
from python.publishing.inventory.assignment_rules import filter_records_available_for_library
from python.publishing.schemas.models import PuzzleRecord


def analyze_book_capacity(
    *,
    spec,
    puzzle_records: List[PuzzleRecord],
    library_inventory: dict | None = None,
) -> Dict:
    base_records = list(puzzle_records)
    if library_inventory is not None:
        base_records = filter_records_available_for_library(
            records=base_records,
            inventory=library_inventory,
        )

    sections_report = []
    total_requested = 0
    total_eligible = 0
    buildable = True

    for section_spec in spec.sections:
        eligible = select_puzzles_for_section(
            puzzle_records=base_records,
            section_criteria=section_spec.criteria,
            global_filters=spec.global_filters,
        )
        eligible_count = len(eligible)
        requested = int(section_spec.puzzle_count)
        shortage = max(0, requested - eligible_count)

        if shortage > 0:
            buildable = False

        sections_report.append(
            {
                "section_code": section_spec.section_code,
                "title": section_spec.title,
                "difficulty_label_hint": section_spec.difficulty_label_hint,
                "requested": requested,
                "eligible": eligible_count,
                "shortage": shortage,
                "sample_record_ids": [record.record_id for record in eligible[:10]],
            }
        )

        total_requested += requested
        total_eligible += eligible_count

    return {
        "book_id": spec.book_id,
        "library_id": spec.library_id,
        "buildable": buildable,
        "total_requested": total_requested,
        "total_eligible_across_sections": total_eligible,
        "sections": sections_report,
    }


def explain_capacity_failure(report: Dict) -> List[str]:
    messages: List[str] = []

    if report.get("buildable", False):
        messages.append("All sections currently have enough eligible puzzles.")
        return messages

    for section in report.get("sections", []):
        shortage = int(section.get("shortage", 0))
        if shortage <= 0:
            continue

        difficulty_hint = section.get("difficulty_label_hint")
        difficulty_text = f" ({difficulty_hint})" if difficulty_hint else ""
        messages.append(
            f"Section {section['section_code']}{difficulty_text} is short by {shortage} puzzle(s): "
            f"requested {section['requested']}, eligible {section['eligible']}."
        )

    messages.append(
        "Action: generate more puzzles matching the missing section criteria, or relax the section filters."
    )
    return messages