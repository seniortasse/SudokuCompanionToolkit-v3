from __future__ import annotations

from typing import Dict, List

from python.publishing.difficulty.labeler import classify_weight, make_effort_label
from python.publishing.schemas.models import PatternRecord, PrintHeader


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def build_app_search_tags(
    *,
    difficulty_label: str,
    difficulty_band_code: str | None,
    technique_count: int,
    techniques_used: List[str],
    pattern: PatternRecord | None,
    layout_type: str,
    clue_count: int,
) -> List[str]:
    tags: List[str] = [
        difficulty_label,
        f"{technique_count}_techniques",
        layout_type,
        f"{clue_count}_clues",
    ]

    if difficulty_band_code:
        tags.append(difficulty_band_code.lower())

    tags.extend(techniques_used)

    if pattern is not None:
        tags.append(pattern.pattern_id.lower())
        tags.append(pattern.slug)
        tags.append(pattern.visual_family)
        tags.extend(pattern.tags)

    return _dedupe_preserve_order([t.lower() for t in tags])


def build_print_header(
    *,
    display_code: str,
    difficulty_label: str,
    weight: int,
) -> PrintHeader:
    return PrintHeader(
        display_code=display_code,
        difficulty_label=difficulty_label.title(),
        effort_label=make_effort_label(weight),
    )


def enrich_puzzle_metadata(
    *,
    weight: int,
    technique_count: int,
    techniques_used: List[str],
    pattern: PatternRecord | None,
    layout_type: str,
    clue_count: int,
    display_code: str,
) -> Dict[str, object]:
    difficulty = classify_weight(weight)

    app_search_tags = build_app_search_tags(
        difficulty_label=difficulty.label,
        difficulty_band_code=difficulty.code,
        technique_count=technique_count,
        techniques_used=techniques_used,
        pattern=pattern,
        layout_type=layout_type,
        clue_count=clue_count,
    )

    print_header = build_print_header(
        display_code=display_code,
        difficulty_label=difficulty.label,
        weight=weight,
    )

    return {
        "difficulty_label": difficulty.label,
        "difficulty_band_code": difficulty.code,
        "app_search_tags": app_search_tags,
        "print_header": print_header,
    }