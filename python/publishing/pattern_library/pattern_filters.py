from __future__ import annotations

from typing import Iterable, List, Optional

from python.publishing.schemas.models import PatternRecord


def filter_patterns(
    patterns: Iterable[PatternRecord],
    *,
    status: Optional[str] = None,
    family_id: Optional[str] = None,
    tag: Optional[str] = None,
    min_clue_count: Optional[int] = None,
    max_clue_count: Optional[int] = None,
    min_aesthetic_score: Optional[float] = None,
    min_print_score: Optional[float] = None,
    min_legibility_score: Optional[float] = None,
) -> List[PatternRecord]:
    out: List[PatternRecord] = []

    for pattern in patterns:
        if status and str(pattern.status) != str(status):
            continue

        if family_id and str(pattern.family_id or "") != str(family_id):
            continue

        if tag and str(tag) not in set(pattern.tags or []):
            continue

        if min_clue_count is not None and int(pattern.clue_count or 0) < int(min_clue_count):
            continue

        if max_clue_count is not None and int(pattern.clue_count or 0) > int(max_clue_count):
            continue

        if min_aesthetic_score is not None:
            score = float(pattern.aesthetic_score) if pattern.aesthetic_score is not None else 0.0
            if score < float(min_aesthetic_score):
                continue

        if min_print_score is not None:
            score = float(pattern.print_score) if pattern.print_score is not None else 0.0
            if score < float(min_print_score):
                continue

        if min_legibility_score is not None:
            score = float(pattern.legibility_score) if pattern.legibility_score is not None else 0.0
            if score < float(min_legibility_score):
                continue

        out.append(pattern)

    return out