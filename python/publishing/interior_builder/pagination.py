from __future__ import annotations

from python.publishing.schemas.models import InteriorPlan
from python.publishing.schemas.page_types import BLANK_PAGE


def apply_page_numbering(plan: InteriorPlan, policy: str) -> InteriorPlan:
    policy = str(policy or "physical_all_suppress_blank_only").strip().lower()

    _clear_numbering_fields(plan)
    _assign_physical_page_numbers(plan)

    if policy == "none":
        for block in plan.page_blocks:
            block.logical_page_number = block.physical_page_number
            block.show_page_number = False
            block.page_number_style = None
        return plan

    if policy in {
        "physical",
        "physical_all",
        "physical_all_suppress_blank_only",
        "physical_book_wide",
        "phase1_constitution",
        "phase1b_full_mirror",
        "physical_all_suppress_cover_blank_section_openers",
    }:
        for block in plan.page_blocks:
            block.logical_page_number = block.physical_page_number
            block.show_page_number = block.page_type != BLANK_PAGE
            block.page_number_style = "arabic" if block.show_page_number else None
            if block.show_page_number:
                block.payload["printed_page_number"] = block.logical_page_number
            else:
                block.payload.pop("printed_page_number", None)
        return plan

    raise ValueError(f"Unknown page numbering policy: {policy}")


def _clear_numbering_fields(plan: InteriorPlan) -> None:
    for block in plan.page_blocks:
        block.payload.pop("printed_page_number", None)
        block.physical_page_number = None
        block.logical_page_number = None


def _assign_physical_page_numbers(plan: InteriorPlan) -> None:
    for index, block in enumerate(plan.page_blocks, start=1):
        block.page_index = index
        block.physical_page_number = index