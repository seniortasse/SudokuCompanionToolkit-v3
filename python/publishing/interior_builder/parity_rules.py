from __future__ import annotations

from typing import Dict, List, Optional, Set

from python.publishing.schemas.models import InteriorPlan, PageBlock
from python.publishing.schemas.page_types import (
    BLANK_PAGE,
    FEATURES_PAGE,
    PROMO_PAGE,
    RULES_PAGE,
    SECTION_HIGHLIGHTS_PAGE,
    SECTION_OPENER_PAGE,
    SECTION_PATTERN_GALLERY_PAGE,
    SOLUTION_PAGE,
    TITLE_PAGE,
    TOC_PAGE,
    TUTORIAL_PAGE,
    WARMUP_PAGE,
    WELCOME_PAGE,
)


_MAJOR_SECTION_START_TYPES = {
    SECTION_OPENER_PAGE,
    SECTION_HIGHLIGHTS_PAGE,
    SECTION_PATTERN_GALLERY_PAGE,
    "SOLUTION_SECTION_OPENER_PAGE",
    SOLUTION_PAGE,
    PROMO_PAGE,
}

_FRONT_MATTER_TYPES = {
    TITLE_PAGE,
    WELCOME_PAGE,
    FEATURES_PAGE,
    TOC_PAGE,
    RULES_PAGE,
    TUTORIAL_PAGE,
    WARMUP_PAGE,
}


def insert_required_blank_pages(
    plan: InteriorPlan,
    blank_page_policy: str,
    *,
    recto_start_policy: Optional[Dict[str, object]] = None,
) -> InteriorPlan:
    explicit_targets = _resolve_explicit_recto_targets(recto_start_policy)

    policy = str(blank_page_policy or "none").strip().lower()
    effective_policy = "explicit_targets" if explicit_targets is not None else policy

    if effective_policy == "none" and policy == "none":
        _reindex_page_blocks(plan)
        plan.requires_blank_page_adjustment = False
        return plan

    original = list(plan.page_blocks)
    updated: List[PageBlock] = []

    for block in original:
        if _needs_recto_start(block, updated, effective_policy, explicit_targets=explicit_targets):
            updated.append(
                PageBlock(
                    page_type=BLANK_PAGE,
                    template_id="blank_page_basic",
                    show_page_number=False,
                    page_number_style=None,
                    payload={
                        "inserted_by_policy": policy,
                        "reason": _blank_reason_for(block),
                    },
                )
            )

        updated.append(block)

    
    if policy == "enforce_recto_section_starts_plus_even_end":
        if len(updated) % 2 == 1:
            updated.append(
                PageBlock(
                    page_type=BLANK_PAGE,
                    template_id="blank_page_basic",
                    show_page_number=False,
                    page_number_style=None,
                    payload={
                        "inserted_by_policy": policy,
                        "reason": "Inserted to make final page count even.",
                    },
                )
            )

    plan.page_blocks = updated
    _reindex_page_blocks(plan)
    plan.requires_blank_page_adjustment = any(
        block.page_type == BLANK_PAGE for block in plan.page_blocks
    )
    return plan


def _resolve_explicit_recto_targets(
    recto_start_policy: Optional[Dict[str, object]],
) -> Optional[Set[str]]:
    data = dict(recto_start_policy or {})
    if not data:
        return None

    enabled = bool(data.get("enabled", False))
    if not enabled:
        return None

    targets_raw = data.get("targets", [])
    if not isinstance(targets_raw, list):
        return None

    targets = {str(x).strip() for x in targets_raw if str(x).strip()}
    if not targets:
        return None

    return targets


def _needs_recto_start(
    block: PageBlock,
    existing_blocks: List[PageBlock],
    policy: str,
    *,
    explicit_targets: Optional[Set[str]] = None,
) -> bool:
    if not existing_blocks:
        return False

    next_physical_index = len(existing_blocks) + 1
    next_is_recto = (next_physical_index % 2) == 1

    if next_is_recto:
        return False

    if policy == "explicit_targets":
        return explicit_targets is not None and block.page_type in explicit_targets

    if policy in {"enforce_recto_section_starts", "enforce_recto_section_starts_plus_even_end"}:
        return block.page_type == SECTION_OPENER_PAGE

    if policy == "enforce_recto_solutions_start":
        return block.page_type == SOLUTION_PAGE

    if policy == "enforce_recto_major_sections":
        return block.page_type in _MAJOR_SECTION_START_TYPES

    if policy == "enforce_recto_front_matter_and_sections":
        return block.page_type in (_FRONT_MATTER_TYPES | _MAJOR_SECTION_START_TYPES)

    return False


def _blank_reason_for(block: PageBlock) -> str:
    if block.page_type == SECTION_OPENER_PAGE:
        return "Inserted to force section opener onto recto page."
    if block.page_type == SECTION_HIGHLIGHTS_PAGE:
        return "Inserted to force section highlights onto recto page."
    if block.page_type == SECTION_PATTERN_GALLERY_PAGE:
        return "Inserted to force section pattern gallery onto recto page."
    if block.page_type == "SOLUTION_SECTION_OPENER_PAGE":
        return "Inserted to force solution section opener onto recto page."
    if block.page_type == SOLUTION_PAGE:
        return "Inserted to force solutions onto recto page."
    if block.page_type == PROMO_PAGE:
        return "Inserted to force end-matter promo onto recto page."
    if block.page_type in _FRONT_MATTER_TYPES:
        return "Inserted to force front matter page onto recto page."
    return "Inserted by blank-page policy."


def _reindex_page_blocks(plan: InteriorPlan) -> None:
    for index, block in enumerate(plan.page_blocks, start=1):
        block.page_index = index
    plan.estimated_page_count = len(plan.page_blocks)